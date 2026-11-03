import json
from contextlib import nullcontext
from pathlib import Path
from threading import Lock

from .tasks import MEDICAL_TASK, build_messages


class QwenEngine:
    def __init__(self, model_dir, adapter_dir=None):
        self.model_dir = Path(model_dir)
        self.adapter_dir = Path(adapter_dir) if adapter_dir is not None else None
        self._model = None
        self._processor = None
        self._adapter_loaded = False
        self._lock = Lock()

    def _check_model_files(self):
        config = self.model_dir / "config.json"
        index = self.model_dir / "model.safetensors.index.json"
        single_file = self.model_dir / "model.safetensors"
        if not config.is_file():
            raise FileNotFoundError("找不到本地模型配置，模型可能尚未下载完成。")
        if index.is_file():
            data = json.loads(index.read_text(encoding="utf-8"))
            shards = set(data.get("weight_map", {}).values())
            if not shards or any(not (self.model_dir / shard).is_file() for shard in shards):
                raise FileNotFoundError("本地模型权重尚未下载完整。")
        elif not single_file.is_file():
            raise FileNotFoundError("找不到完整的本地模型权重，下载可能仍在进行。")

    def _load(self):
        if self._model is not None:
            return
        self._check_model_files()

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用的 CUDA 显卡，请检查 PyTorch 和显卡驱动。")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        self._processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            quantization_config=quantization,
            torch_dtype=compute_dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()

    def _check_adapter_files(self):
        if self.adapter_dir is None or any(
            not (self.adapter_dir / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")
        ):
            raise FileNotFoundError("找不到完整的医学影像微调适配器。")

    def _load_adapter(self):
        if self._adapter_loaded:
            return
        self._check_adapter_files()
        from peft import PeftModel

        self._model = PeftModel.from_pretrained(self._model, str(self.adapter_dir), is_trainable=False).eval()
        self._adapter_loaded = True

    def infer(self, task, media_path, question):
        messages = build_messages(task, media_path, question)
        with self._lock:
            if task == MEDICAL_TASK:
                self._check_adapter_files()
            self._load()
            if task == MEDICAL_TASK:
                self._load_adapter()
            import torch
            from qwen_vl_utils import process_vision_info

            if task == "视频事件定位":
                from .video import make_timed_frame_messages, sample_video

                frames, sample_fps, duration = sample_video(media_path, max_frames=8)
                messages = make_timed_frame_messages(
                    frames, sample_fps, duration, messages[0]["content"][1]["text"]
                )

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self._processor(
                text=[text], images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs
            ).to("cuda")
            adapter_context = self._model.disable_adapter() if self._adapter_loaded and task != MEDICAL_TASK else nullcontext()
            with adapter_context, torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            response = self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            if task == "视频事件定位":
                response += f"\n\n视频实际时长：{duration:.1f} 秒。模型时间为抽帧估计，请对照原视频核对。"
            return response
