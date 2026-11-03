import os
from pathlib import Path

import gradio as gr

from vl_workbench.progress import read_training_progress


ROOT = Path(__file__).resolve().parent
RUN = os.environ.get("QWEN_TRAIN_RUN", "pneumonia_qlora_highres")
OUTPUT = ROOT / "outputs" / RUN


def status():
    return read_training_progress(OUTPUT)


with gr.Blocks(title="医学影像微调进度") as monitor:
    gr.Markdown(
        "# 医学影像微调进度\n"
        "Qwen2.5-VL-3B · LLaMA-Factory · PneumoniaMNIST\n\n"
        "256 张均衡训练图像，保留官方验证集和测试集。状态来自本地训练日志，每 10 秒刷新。"
    )
    progress = gr.Textbox(label=f"训练记录：{RUN}", value=status(), lines=5, interactive=False)
    gr.Markdown("训练占用显存期间，请等待训练结束后再运行主界面中的模型推理。")
    timer = gr.Timer(10)
    timer.tick(status, outputs=progress)


if __name__ == "__main__":
    monitor.launch(server_name="127.0.0.1", server_port=int(os.environ.get("QWEN_MONITOR_PORT", "7861")), share=False)
