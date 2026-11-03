import tempfile
import unittest
from pathlib import Path

from vl_workbench.tasks import build_messages, parse_boxes
from scripts.prepare_dataset import NUMERIC_PROMPT as TRAIN_PROMPT


class TaskMessageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.media = Path(self.temp.name) / "example.png"
        self.media.write_bytes(b"test")

    def tearDown(self):
        self.temp.cleanup()

    def test_ocr_message_contains_image_and_instruction(self):
        messages = build_messages("OCR 文字识别", str(self.media), "")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["image"], str(self.media))
        self.assertIn("识别", content[1]["text"])

    def test_video_message_includes_fps_and_user_question(self):
        messages = build_messages("视频事件定位", str(self.media), "什么时候开门？")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "video")
        self.assertGreater(content[0]["fps"], 0)
        self.assertIn("什么时候开门？", content[1]["text"])

    def test_missing_input_fails_without_attempting_inference(self):
        with self.assertRaisesRegex(ValueError, "上传"):
            build_messages("图片阅读理解", None, "")

    def test_unknown_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "任务"):
            build_messages("unknown", str(self.media), "")

    def test_medical_mode_matches_training_prompt_and_resolution(self):
        content = build_messages("医学影像分类（微调模型）", str(self.media), "")[0]["content"]
        self.assertEqual(content[1]["text"], TRAIN_PROMPT.replace("<image>", "", 1))
        self.assertEqual(content[0]["min_pixels"], 3136)
        self.assertEqual(content[0]["max_pixels"], 50176)


class BoundingBoxTests(unittest.TestCase):
    def test_reads_fenced_json_and_clips_to_image(self):
        response = '```json\n[{"bbox_2d": [-5, 2, 130, 50], "label": "cat"}]\n```'
        self.assertEqual(parse_boxes(response, 100, 80), [([0, 2, 100, 50], "cat")])

    def test_ignores_invalid_or_degenerate_boxes(self):
        response = '[{"bbox_2d": [3, 3, 3, 10]}, {"bbox_2d": ["a", 0, 1, 2]}]'
        self.assertEqual(parse_boxes(response, 100, 80), [])

    def test_non_json_response_is_not_interpreted_as_a_box(self):
        self.assertEqual(parse_boxes("没有找到目标", 100, 80), [])

    def test_rescales_coordinates_from_model_input_to_original_image(self):
        response = '[{"bbox_2d": [59, 348, 197, 378], "label": "total"}]'
        self.assertEqual(
            parse_boxes(response, 1000, 620, reference_width=784, reference_height=476),
            [([75, 453, 251, 492], "total")],
        )


if __name__ == "__main__":
    unittest.main()
