import random
import re
from collections import Counter
from pathlib import Path


CLASSES = ("NORMAL", "PNEUMONIA")


def parse_label(text):
    digit = re.fullmatch(r"\s*([01])\s*[.!]?\s*", str(text))
    if digit:
        return CLASSES[int(digit.group(1))]
    matches = set(re.findall(r"\b(?:NORMAL|PNEUMONIA)\b", str(text).upper()))
    return matches.pop() if len(matches) == 1 else None


def calculate_metrics(labels, predictions):
    if not labels or len(labels) != len(predictions) or any(label not in CLASSES for label in labels):
        raise ValueError("Labels and predictions must have matching non-empty lengths and valid labels.")
    counts = Counter(labels)
    correct = Counter(label for label, prediction in zip(labels, predictions) if label == prediction)
    if any(counts[label] == 0 for label in CLASSES):
        raise ValueError("Both classes are required for balanced accuracy.")
    recall = {label: correct[label] / counts[label] for label in CLASSES}
    return {
        "samples": len(labels),
        "accuracy": sum(correct.values()) / len(labels),
        "balanced_accuracy": sum(recall.values()) / len(CLASSES),
        "recall_by_class": recall,
        "unparseable": sum(prediction not in CLASSES for prediction in predictions),
    }


def select_balanced(rows, per_class, seed=42, excluded_images=()):
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    rng = random.Random(seed)
    selected = []
    for label in CLASSES:
        group = [
            row for row in rows
            if row["label"] == label and (not excluded_images or Path(row["image"]).name not in excluded_images)
        ]
        if len(group) < per_class:
            raise ValueError(f"Not enough {label} examples")
        selected.extend(rng.sample(group, per_class))
    rng.shuffle(selected)
    return selected
