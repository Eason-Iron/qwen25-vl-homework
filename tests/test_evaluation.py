import unittest

from vl_workbench.evaluation import calculate_metrics, parse_label, select_balanced


class EvaluationTests(unittest.TestCase):
    def test_unexpected_or_ambiguous_output_is_not_counted_as_correct(self):
        self.assertEqual(parse_label("NORMAL"), "NORMAL")
        self.assertEqual(parse_label("PNEUMONIA."), "PNEUMONIA")
        self.assertEqual(parse_label("0"), "NORMAL")
        self.assertEqual(parse_label("1."), "PNEUMONIA")
        self.assertIsNone(parse_label("0 or 1"))
        self.assertIsNone(parse_label("NORMAL or PNEUMONIA"))
        self.assertIsNone(parse_label("unable to decide"))

    def test_metrics_count_unparseable_predictions_as_incorrect(self):
        metrics = calculate_metrics(["NORMAL", "NORMAL", "PNEUMONIA", "PNEUMONIA"], ["NORMAL", None, "NORMAL", "PNEUMONIA"])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["unparseable"], 1)

    def test_balanced_selection_is_reproducible(self):
        rows = [{"label": label, "index": i} for label in ("NORMAL", "PNEUMONIA") for i in range(10)]
        sample = select_balanced(rows, per_class=3, seed=42)
        self.assertEqual(sample, select_balanced(rows, per_class=3, seed=42))
        self.assertEqual(sum(row["label"] == "NORMAL" for row in sample), 3)
        self.assertEqual(sum(row["label"] == "PNEUMONIA" for row in sample), 3)

    def test_balanced_selection_excludes_previous_test_images(self):
        rows = [
            {"label": label, "image": f"data/test/{label}-{i}.png"}
            for label in ("NORMAL", "PNEUMONIA") for i in range(4)
        ]
        sample = select_balanced(rows, per_class=3, seed=42, excluded_images={"NORMAL-0.png", "PNEUMONIA-0.png"})
        self.assertEqual(len(sample), 6)
        self.assertTrue(all(not row["image"].endswith("-0.png") for row in sample))
        with self.assertRaises(ValueError):
            select_balanced(rows, per_class=4, excluded_images={"NORMAL-0.png"})


if __name__ == "__main__":
    unittest.main()
