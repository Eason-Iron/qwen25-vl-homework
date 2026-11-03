import logging
import os
import json
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from vl_workbench.engine import QwenEngine
from vl_workbench.evaluation import parse_label
from vl_workbench.progress import read_training_progress
from vl_workbench.tasks import MAX_IMAGE_PIXELS, MEDICAL_TASK, TASK_PROMPTS, parse_boxes


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("QWEN_MODEL_DIR", ROOT / "models" / "Qwen2.5-VL-3B-Instruct"))
ADAPTER_DIR = Path(os.environ.get("QWEN_MEDICAL_ADAPTER", ROOT / "adapters" / "pneumonia-qlora"))
ENGINE = QwenEngine(MODEL_DIR, ADAPTER_DIR)

CSS = """
:root { --ink: #182f35; --accent: #147c70; --paper: #f5f7f3; }
.gradio-container { max-width: 1380px !important; background: var(--paper) !important; color: var(--ink); }
.hero { background: #173840; color: #f5f8f2; border-radius: 18px; padding: 26px 32px; margin: 8px 0 18px; }
.hero .eyebrow { color: #8fd3c4; letter-spacing: .19em; font-size: 11px; font-weight: 700; }
.hero h1 { color: #fff; font-family: Georgia, 'Microsoft YaHei', serif; font-size: 34px; margin: 8px 0 6px; }
.hero p { color: #d1e3dc; margin: 0; font-size: 14px; }
.panel-title { font-size: 12px; letter-spacing: .12em; font-weight: 700; color: #42766d; margin-bottom: 8px; }
.hint { color: #58716b; font-size: 13px; line-height: 1.7; }
#run-button { background: var(--accent); color: white; border: none; min-height: 48px; font-weight: 700; }
footer { display: none !important; }
"""


def select_media(task):
    video = task == "视频事件定位"
    medical = task == MEDICAL_TASK
    return (
        gr.update(visible=not video), gr.update(visible=video),
        gr.update(visible=not medical, value=""), gr.update(visible=medical),
    )


def load_ocr_example():
    example = ROOT / "examples" / "ocr-example.png"
    if not example.is_file():
        raise gr.Error("找不到示例图片，请先运行 scripts/create_ocr_example.py。")
    return "OCR 文字识别", gr.update(value=str(example), visible=True), gr.update(visible=False)


def load_video_example():
    example = ROOT / "examples" / "bunny-clip.mp4"
    if not example.is_file():
        example = ROOT / "examples" / "space-clip.mp4"
    if not example.is_file():
        raise gr.Error("未找到本地示例视频，请上传视频。")
    return "视频事件定位", gr.update(visible=False), gr.update(value=str(example), visible=True)


def load_image_example(task, filename):
    example = ROOT / "examples" / filename
    if not example.is_file():
        raise gr.Error("未找到本地示例图片，请上传图片。")
    return task, gr.update(value=str(example), visible=True), gr.update(visible=False)


def load_medical_example():
    test_file = ROOT / "data" / "processed_numeric" / "test.jsonl"
    if not test_file.is_file():
        raise gr.Error("请先准备 PneumoniaMNIST 数据。")
    for line in test_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if parse_label(row["conversations"][1]["value"]) == "PNEUMONIA":
            return MEDICAL_TASK, gr.update(value=row["images"][0], visible=True), gr.update(visible=False)
    raise gr.Error("测试数据中没有找到肺炎样例。")


def annotate_image(image_path, response):
    if not image_path:
        return None
    image = Image.open(image_path).convert("RGB")
    from qwen_vl_utils.vision_process import fetch_image

    processed = fetch_image({"image": image, "max_pixels": MAX_IMAGE_PIXELS})
    boxes = parse_boxes(
        response, image.width, image.height, reference_width=processed.width, reference_height=processed.height
    )
    if not boxes:
        return None
    draw = ImageDraw.Draw(image)
    for box, _label in boxes:
        draw.rectangle(box, outline="#00c29a", width=max(2, round(min(image.size) / 250)))
    return image


def run_task(task, image_path, video_path, question):
    media = video_path if task == "视频事件定位" else image_path
    try:
        response = ENGINE.infer(task, media, question)
        if task == MEDICAL_TASK:
            predicted = parse_label(response)
            label = {"NORMAL": "NORMAL（正常）", "PNEUMONIA": "PNEUMONIA（肺炎）"}.get(predicted, "无法解析")
            response = f"模型原始输出：{response}\n预测类别：{label}\n\n仅供课程实验，不能用于临床诊断。"
        annotated = annotate_image(image_path, response) if task == "视觉定位" else None
        return response, annotated
    except Exception as exc:
        logging.exception("Inference failed")
        return f"运行未完成：{exc}", None


def show_training_progress(run):
    folder = {"试运行": "pneumonia_smoke", "第一版": "pneumonia_qlora", "较高分辨率版": "pneumonia_qlora_highres", "数字标签版": "pneumonia_qlora_numeric"}[run]
    return read_training_progress(ROOT / "outputs" / folder)


def build_app():
    with gr.Blocks(title="Qwen2.5-VL 多模态实验", css=CSS) as app:
        gr.HTML(
            '<div class="hero"><div class="eyebrow">WEEK 07 · MULTIMODAL LAB</div>'
            '<h1>让模型读懂图像与视频</h1>'
            '<p>Qwen2.5-VL 实验工作台 / 本地推理 / 六类视觉任务</p></div>'
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.HTML('<div class="panel-title">01 / 选择实验与输入</div>')
                task = gr.Radio(list(TASK_PROMPTS), value="OCR 文字识别", label="实验任务")
                image_input = gr.Image(type="filepath", label="上传图片", sources=["upload", "clipboard"])
                video_input = gr.Video(label="上传视频", visible=False, sources=["upload"])
                with gr.Row():
                    example_button = gr.Button("载入示例票据", size="sm")
                    video_example_button = gr.Button("载入示例视频", size="sm")
                with gr.Row():
                    scenery_example_button = gr.Button("载入风景示例", size="sm")
                    person_example_button = gr.Button("载入公众人物示例", size="sm")
                medical_example_button = gr.Button("载入医学影像示例", size="sm")
                medical_hint = gr.HTML(
                    '<p class="hint">医学示例选自测试集的肺炎类别；标签不会发送给模型，预测可能错误。</p>',
                    visible=False,
                )
                question = gr.Textbox(
                    label="补充问题或定位目标",
                    placeholder="例如：图中表格的合计是多少？或：找到红色车辆",
                    lines=2,
                )
                run = gr.Button("开始分析", elem_id="run-button", variant="primary")
                gr.HTML('<p class="hint">首次运行需加载本地模型，可能需要一些时间。视频建议先使用短片。</p>')
            with gr.Column(scale=6):
                gr.HTML('<div class="panel-title">02 / 模型输出</div>')
                output = gr.Textbox(label="分析结果", lines=14, interactive=False, show_copy_button=True)
                visual = gr.Image(label="视觉定位结果", type="pil", interactive=False)
                gr.HTML('<p class="hint">绿色方框根据模型返回的坐标绘制，请结合原图核对。所有结果仅供课程实验。</p>')
        with gr.Accordion("医学影像微调进度", open=True):
            run_choice = gr.Radio(["试运行", "第一版", "较高分辨率版", "数字标签版"], value="数字标签版", label="训练记录")
            progress = gr.Textbox(label="实际日志", value=show_training_progress("数字标签版"), lines=4, interactive=False)
            refresh = gr.Button("刷新训练进度", size="sm")
        task.change(select_media, inputs=task, outputs=[image_input, video_input, question, medical_hint])
        example_button.click(load_ocr_example, outputs=[task, image_input, video_input])
        video_example_button.click(load_video_example, outputs=[task, image_input, video_input])
        scenery_example_button.click(
            lambda: load_image_example("风景识别", "scenery.jpg"), outputs=[task, image_input, video_input]
        )
        person_example_button.click(
            lambda: load_image_example("公众人物识别", "public-figure.jpg"), outputs=[task, image_input, video_input]
        )
        medical_example_button.click(load_medical_example, outputs=[task, image_input, video_input])
        run_choice.change(show_training_progress, inputs=run_choice, outputs=progress)
        refresh.click(show_training_progress, inputs=run_choice, outputs=progress)
        run.click(run_task, inputs=[task, image_input, video_input, question], outputs=[output, visual])
    return app


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860, share=False)
