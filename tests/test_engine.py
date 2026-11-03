import tempfile
import unittest
from pathlib import Path

from vl_workbench.engine import QwenEngine


class EnginePreflightTests(unittest.TestCase):
    def test_missing_model_is_reported_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "image.png"
            media.write_bytes(b"sample")
            engine = QwenEngine(Path(directory) / "missing-model")
            with self.assertRaisesRegex(FileNotFoundError, "模型"):
                engine.infer("OCR 文字识别", str(media), "")

    def test_missing_media_is_reported_before_loading_model(self):
        engine = QwenEngine("missing-model")
        with self.assertRaisesRegex(ValueError, "上传"):
            engine.infer("OCR 文字识别", None, "")

    def test_missing_medical_adapter_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = QwenEngine("missing-model", Path(directory) / "missing-adapter")
            with self.assertRaisesRegex(FileNotFoundError, "微调"):
                engine._check_adapter_files()


if __name__ == "__main__":
    unittest.main()
