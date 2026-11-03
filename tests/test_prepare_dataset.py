import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.prepare_dataset import prepare_dataset


class DatasetPreparationTests(unittest.TestCase):
    def test_preserves_splits_and_balances_limited_training_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.npz"
            np.savez_compressed(
                source,
                train_images=np.arange(16 * 6, dtype=np.uint8).reshape(6, 4, 4),
                train_labels=np.array([[0], [0], [0], [1], [1], [1]], dtype=np.uint8),
                val_images=np.zeros((1, 4, 4), dtype=np.uint8),
                val_labels=np.array([[0]], dtype=np.uint8),
                test_images=np.ones((1, 4, 4), dtype=np.uint8),
                test_labels=np.array([[1]], dtype=np.uint8),
            )
            result = prepare_dataset(source, root / "processed", train_limit=4, seed=42)
            self.assertEqual(result["train"], {"NORMAL": 2, "PNEUMONIA": 2})
            for split, count in (("train", 4), ("val", 1), ("test", 1)):
                rows = [json.loads(line) for line in (root / "processed" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), count)
                for row in rows:
                    self.assertIn("<image>", row["conversations"][0]["value"])
                    self.assertTrue(Path(row["images"][0]).is_file())
                    self.assertIn(split, Path(row["images"][0]).parts)

    def test_rejects_unknown_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.npz"
            data = {}
            for split in ("train", "val", "test"):
                data[f"{split}_images"] = np.zeros((1, 4, 4), dtype=np.uint8)
                data[f"{split}_labels"] = np.array([[2]], dtype=np.uint8)
            np.savez_compressed(source, **data)
            with self.assertRaisesRegex(ValueError, "label"):
                prepare_dataset(source, root / "processed")

    def test_numeric_labels_preserve_class_meaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.npz"
            values = {}
            for split in ("train", "val", "test"):
                values[f"{split}_images"] = np.zeros((2, 4, 4), dtype=np.uint8)
                values[f"{split}_labels"] = np.array([[0], [1]], dtype=np.uint8)
            np.savez_compressed(source, **values)
            summary = prepare_dataset(source, root / "numeric", label_mode="numeric")
            rows = [json.loads(line) for line in (root / "numeric" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["test"], {"NORMAL": 1, "PNEUMONIA": 1})
            self.assertEqual([row["conversations"][1]["value"] for row in rows], ["0", "1"])
            self.assertIn("0 for NORMAL, 1 for PNEUMONIA", rows[0]["conversations"][0]["value"])


if __name__ == "__main__":
    unittest.main()
