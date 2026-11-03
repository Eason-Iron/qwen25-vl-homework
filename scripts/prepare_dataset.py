import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
PROMPT = (
    "<image>Classify this pediatric chest X-ray in the PneumoniaMNIST dataset as "
    "NORMAL or PNEUMONIA. Reply with exactly one label."
)
NUMERIC_PROMPT = (
    "<image>Classify this pediatric chest X-ray in the PneumoniaMNIST dataset. "
    "Reply with exactly one digit: 0 for NORMAL, 1 for PNEUMONIA."
)
EXPECTED_MD5 = "8f4eceb4ccffa70c672198ea285246c6"


def prepare_dataset(source, output_dir, train_limit=None, seed=42, label_mode="name"):
    if label_mode not in {"name", "numeric"}:
        raise ValueError("Unknown label mode")
    output_dir = Path(output_dir)
    with np.load(source, allow_pickle=False) as data:
        splits = {}
        for split in ("train", "val", "test"):
            images = data[f"{split}_images"]
            labels = data[f"{split}_labels"].reshape(-1)
            if len(images) != len(labels) or images.ndim != 3:
                raise ValueError(f"Invalid image or label shape for {split}")
            if any(int(label) not in LABELS or int(label) != label for label in labels):
                raise ValueError(f"Invalid label in {split}")
            splits[split] = (images, labels)

    if train_limit is not None and train_limit <= 0:
        raise ValueError("train_limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, (images, labels) in splits.items():
        indices = list(range(len(labels)))
        if split == "train":
            rng = np.random.default_rng(seed)
            groups = [list(rng.permutation(np.flatnonzero(labels == label))) for label in LABELS]
            indices = []
            while any(groups):
                for group in groups:
                    if group:
                        indices.append(int(group.pop()))
            if train_limit is not None:
                indices = indices[:train_limit]
        image_dir = output_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        split_counts = Counter()
        for index in indices:
            label = LABELS[int(labels[index])]
            image_path = image_dir / f"{index:05d}.png"
            Image.fromarray(images[index]).convert("RGB").save(image_path)
            rows.append({
                "conversations": [
                    {"from": "human", "value": NUMERIC_PROMPT if label_mode == "numeric" else PROMPT},
                    {"from": "gpt", "value": str(int(labels[index])) if label_mode == "numeric" else label},
                ],
                "images": [str(image_path.resolve())],
            })
            split_counts[label] += 1
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = {label: split_counts[label] for label in LABELS.values()}

    dataset_info = {
        f"pneumonia_{split}": {
            "file_name": f"{split}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        }
        for split in ("train", "val", "test")
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Prepare the official PneumoniaMNIST 64px splits.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-mode", choices=("name", "numeric"), default="name")
    args = parser.parse_args()
    digest = hashlib.md5(args.source.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise ValueError("PneumoniaMNIST checksum mismatch; download the official 64px file again.")
    print(json.dumps(prepare_dataset(args.source, args.output_dir, args.train_limit, args.seed, args.label_mode), indent=2))


if __name__ == "__main__":
    main()
