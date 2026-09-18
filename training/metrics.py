"""Metric factory shared by all three trainers."""

from torchmetrics import AUROC, Accuracy, F1Score, Precision


def build_metrics(num_class: int) -> dict:
    kwargs = dict(task="multiclass", num_classes=num_class)
    return {
        "accuracy": Accuracy(**kwargs),
        "f1_score": F1Score(**kwargs, average="macro"),
        "precision": Precision(**kwargs, average="macro"),
        "auroc": AUROC(**kwargs),
    }
