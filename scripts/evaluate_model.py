import argparse
import json
from pathlib import Path

from vl_workbench.evaluation import calculate_metrics, parse_label, select_balanced


def evaluate(model, processor, rows):
    import torch
    from PIL import Image

    results = []
    for index, row in enumerate(rows, start=1):
        image = Image.open(row["image"]).convert("RGB")
        prompt = row["prompt"].replace("<image>", "", 1)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        answer = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        results.append({"image": Path(row["image"]).name, "label": row["label"], "prediction": parse_label(answer), "raw": answer})
        print(f"{index}/{len(rows)}", flush=True)
    metrics = calculate_metrics([row["label"] for row in results], [row["prediction"] for row in results])
    return {"metrics": metrics, "predictions": results}


def main():
    parser = argparse.ArgumentParser(description="Compare the base model and a LoRA adapter on held-out test images.")
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test-data", default="data/processed/test.jsonl")
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--exclude-results", help="Exclude image names from a previous evaluation JSON")
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    rows = []
    for line in Path(args.test_data).read_text(encoding="utf-8").splitlines():
        example = json.loads(line)
        rows.append({
            "image": example["images"][0],
            "prompt": example["conversations"][0]["value"],
            "label": parse_label(example["conversations"][1]["value"]),
        })
    excluded_images = set()
    if args.exclude_results:
        previous = json.loads(Path(args.exclude_results).read_text(encoding="utf-8"))
        excluded_images = {prediction["image"] for prediction in previous["base"]["predictions"]}
    rows = select_balanced(rows, per_class=args.per_class, seed=args.seed, excluded_images=excluded_images)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels, local_files_only=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quantization, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", local_files_only=True,
    ).eval()
    print("Evaluating the base model", flush=True)
    base = evaluate(model, processor, rows)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    print("Evaluating the fine-tuned adapter", flush=True)
    adapted = evaluate(model, processor, rows)
    output = {
        "seed": args.seed, "per_class": args.per_class, "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels, "excluded_images": sorted(excluded_images), "base": base, "adapter": adapted,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"base": base["metrics"], "adapter": adapted["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
