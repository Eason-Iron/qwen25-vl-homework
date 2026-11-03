import json
import math
from pathlib import Path


MAX_IMAGE_PIXELS = 512 * 28 * 28
MEDICAL_TASK = "医学影像分类（微调模型）"
MEDICAL_PROMPT = (
    "Classify this pediatric chest X-ray in the PneumoniaMNIST dataset. "
    "Reply with exactly one digit: 0 for NORMAL, 1 for PNEUMONIA."
)

TASK_PROMPTS = {
    "OCR 文字识别": "请逐行识别图片中的文字，尽量保留原文和换行；不确定的文字请说明。",
    "图片阅读理解": "请阅读图片并回答用户的问题；没有问题时描述图片中的关键信息。",
    "视觉定位": (
        "定位用户指定的目标，输出 JSON 数组，每个元素包含 label 和 "
        "bbox_2d（左上角 x、y 和右下角 x、y，单位为模型输入图像像素）。"
        "如果没有指定目标，就定位图片中的主要物体。不要编造不存在的目标。"
    ),
    "风景识别": "请识别图片中的风景或地标，并说明判断依据；不确定时请明确说明。",
    "公众人物识别": "请尝试识别图片中的公众人物并简要说明；不确定时请明确说明。",
    "视频事件定位": (
        "请根据视频寻找用户描述的事件，尽可能给出开始和结束时间（分:秒）并简述事件；"
        "如果用户没有指定事件，请列出视频中的主要事件。无法确定时请说明。"
    ),
    MEDICAL_TASK: MEDICAL_PROMPT,
}


def build_messages(task, media_path, question):
    if task not in TASK_PROMPTS:
        raise ValueError("未知任务。")
    if not media_path or not Path(media_path).is_file():
        raise ValueError("请先上传对应的图片或视频。")

    prompt = TASK_PROMPTS[task]
    if task != MEDICAL_TASK and question and question.strip():
        prompt += "\n用户的问题：" + question.strip()
    if task == "视频事件定位":
        visual = {"type": "video", "video": str(media_path), "fps": 0.5, "max_pixels": 128 * 28 * 28}
    elif task == MEDICAL_TASK:
        visual = {"type": "image", "image": str(media_path), "min_pixels": 3136, "max_pixels": 50176}
    else:
        visual = {"type": "image", "image": str(media_path), "max_pixels": MAX_IMAGE_PIXELS}
    return [{"role": "user", "content": [visual, {"type": "text", "text": prompt}]}]


def parse_boxes(response, width, height, reference_width=None, reference_height=None):
    if width <= 0 or height <= 0:
        return []
    if reference_width is None:
        reference_width = width
    if reference_height is None:
        reference_height = height
    if reference_width <= 0 or reference_height <= 0:
        return []
    scale_x = width / reference_width
    scale_y = height / reference_height
    decoder = json.JSONDecoder()
    for i, char in enumerate(response):
        if char not in "[{":
            continue
        try:
            data, _ = decoder.raw_decode(response[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        boxes = []
        for item in data:
            if not isinstance(item, dict):
                continue
            coords = item.get("bbox_2d")
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coords):
                continue
            x1 = max(0, min(width, round(coords[0] * scale_x)))
            y1 = max(0, min(height, round(coords[1] * scale_y)))
            x2 = max(0, min(width, round(coords[2] * scale_x)))
            y2 = max(0, min(height, round(coords[3] * scale_y)))
            if x1 < x2 and y1 < y2:
                boxes.append(([x1, y1, x2, y2], str(item.get("label", "目标"))))
        return boxes
    return []
