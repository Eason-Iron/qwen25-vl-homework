import json
from pathlib import Path


def read_training_progress(output_dir):
    output_dir = Path(output_dir)
    log = output_dir / "trainer_log.jsonl"
    if not log.is_file():
        return "尚未开始，或训练日志尚未生成。"
    entry = None
    last_loss = None
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "current_steps" in candidate:
            entry = candidate
            if "loss" in candidate:
                last_loss = candidate["loss"]
    if entry is None:
        return "训练日志已创建，尚无完整的进度记录。"
    status = "已完成" if (output_dir / "train_results.json").is_file() and (output_dir / "adapter_model.safetensors").is_file() else "进行中或已中断"
    return (
        f"状态：{status}\n"
        f"训练步骤：{entry.get('current_steps', '?')} / {entry.get('total_steps', '?')}\n"
        f"最近损失：{last_loss if last_loss is not None else '暂无'}\n"
        f"已用时间：{entry.get('elapsed_time', '暂无')}"
    )
