from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT = Path("C:/Windows/Fonts/msyh.ttc")


def main():
    if not FONT.is_file():
        raise FileNotFoundError("需要 Windows 微软雅黑字体，或自行提供可显示中文的字体。")
    image = Image.new("RGB", (1000, 620), "#f9faf7")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(str(FONT), 44)
    body = ImageFont.truetype(str(FONT), 32)
    draw.text((70, 55), "多模态课程实验 · 示例票据", fill="#16363a", font=title)
    draw.line((70, 130, 930, 130), fill="#168375", width=3)
    lines = ["日期：2026 年 11 月 3 日", "项目：实验材料", "单价：12 元", "数量：3 件", "合计：36 元"]
    for index, line in enumerate(lines):
        draw.text((75, 165 + index * 72), line, fill="#263a39", font=body)
    destination = ROOT / "examples" / "ocr-example.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
