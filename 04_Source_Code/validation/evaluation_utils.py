from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return overall, per-class and confusion-matrix metrics."""

    if labels is None:
        labels = sorted(pd.Series(y_true).dropna().unique().tolist())

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels,
            average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=labels,
            average="weighted", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, labels=labels,
            average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=labels,
            average="macro", zero_division=0
        ),
    }

    report = cast(
        dict[str, dict[str, float]],
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    )

    class_metrics = pd.DataFrame(
        {label: report[label] for label in labels}
    ).T

    class_metrics.index.name = "class"

    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    return summary, class_metrics, confusion