import json
import tempfile
import unittest
from pathlib import Path

from vl_workbench.progress import read_training_progress


class TrainingProgressTests(unittest.TestCase):
    def test_reports_missing_training_run(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIn("尚未开始", read_training_progress(Path(directory)))

    def test_reads_last_complete_log_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {"current_steps": 2, "total_steps": 8, "loss": 0.4, "elapsed_time": "0:01:00"}
            (root / "trainer_log.jsonl").write_text(json.dumps(entry) + "\n{", encoding="utf-8")
            text = read_training_progress(root)
            self.assertIn("2 / 8", text)
            self.assertIn("0.4", text)
            self.assertNotIn("已完成", text)

    def test_final_summary_keeps_most_recent_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"current_steps": 2, "total_steps": 2, "loss": 0.15},
                {"current_steps": 2, "total_steps": 2},
            ]
            (root / "trainer_log.jsonl").write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            (root / "train_results.json").write_text("{}", encoding="utf-8")
            (root / "adapter_model.safetensors").write_bytes(b"test")
            text = read_training_progress(root)
            self.assertIn("已完成", text)
            self.assertIn("0.15", text)


if __name__ == "__main__":
    unittest.main()
