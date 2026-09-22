"""
04_explainability.py

SHAP-based explainability for the XGBoost current-state classification
benchmark.

Important:
The model explains reconstruction of the contemporaneous Efficiency_Status
label. It is not a future-risk or causal explanation system.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

matplotlib.use("Agg")

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


SOURCE_DIR = Path(__file__).resolve().parent

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


from config import (
    FEATURE_DATA_PATH,
    FIGURES_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    ensure_output_directories,
)


def normalize_shap_output(
    shap_values,
    class_count: int,
) -> list[np.ndarray]:
    """
    Normalize SHAP multiclass output across SHAP/XGBoost versions.

    Possible representations include:
      (classes, rows, features)
      (rows, features, classes)
      (rows, features)
    """

    if isinstance(shap_values, list):
        return [
            np.asarray(values)
            for values in shap_values
        ]

    values = np.asarray(
        shap_values
    )

    if (
        values.ndim == 3
        and values.shape[0] == class_count
    ):
        return [
            values[index]
            for index in range(
                class_count
            )
        ]

    if (
        values.ndim == 3
        and values.shape[-1] == class_count
    ):
        return [
            values[:, :, index]
            for index in range(
                class_count
            )
        ]

    if values.ndim == 2:
        return [
            values
        ]

    raise ValueError(
        "Unsupported SHAP output shape: "
        f"{values.shape}"
    )


def main() -> None:
    ensure_output_directories()

    # --------------------------------------------------------------
    # Load model metadata
    # --------------------------------------------------------------
    metadata_path = (
        MODEL_DIR
        / "model_meta.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    feature_columns = metadata[
        "feature_columns"
    ]

    class_names = metadata[
        "class_names"
    ]

    mode_columns = metadata[
        "mode_columns"
    ]

    # --------------------------------------------------------------
    # Load XGBoost using native serialization
    # --------------------------------------------------------------
    model = XGBClassifier()

    model.load_model(
        MODEL_DIR
        / "xgboost_model.json"
    )

    # --------------------------------------------------------------
    # Load and encode data
    # --------------------------------------------------------------
    df = pd.read_parquet(
        FEATURE_DATA_PATH
    )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"]
    )

    df = (
        df.sort_values("Datetime")
        .reset_index(drop=True)
    )

    encoded = pd.get_dummies(
        df,
        columns=["Operation_Mode"],
        prefix="Mode",
        dtype=int,
    )

    for column in mode_columns:
        if column not in encoded.columns:
            encoded[column] = 0

    # --------------------------------------------------------------
    # Sample for SHAP runtime
    # --------------------------------------------------------------
    sample_size = min(
        3000,
        len(encoded),
    )

    sample = encoded.sample(
        n=sample_size,
        random_state=42,
    )

    X_sample = sample[
        feature_columns
    ]

    # --------------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------------
    print(
        "=" * 72
    )

    print(
        "XGBOOST SHAP EXPLAINABILITY"
    )

    print(
        "=" * 72
    )

    print(
        f"Sample rows: {len(X_sample):,}"
    )

    explainer = shap.TreeExplainer(
        model
    )

    raw_shap_values = (
        explainer.shap_values(
            X_sample
        )
    )

    shap_per_class = (
        normalize_shap_output(
            raw_shap_values,
            len(class_names),
        )
    )

    # --------------------------------------------------------------
    # Global mean absolute SHAP
    # --------------------------------------------------------------
    if len(shap_per_class) == 1:
        mean_absolute_shap = (
            np.abs(
                shap_per_class[0]
            )
            .mean(axis=0)
        )
    else:
        mean_absolute_shap = np.mean(
            [
                np.abs(values)
                .mean(axis=0)
                for values
                in shap_per_class
            ],
            axis=0,
        )

    importance = pd.Series(
        mean_absolute_shap,
        index=feature_columns,
        name="mean_abs_shap",
    ).sort_values(
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR
        / "shap_global_importance.csv"
    )

    print(
        "\nTop SHAP features:"
    )

    print(
        importance
        .head(10)
        .to_string()
    )

    # --------------------------------------------------------------
    # Global importance figure
    # --------------------------------------------------------------
    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    (
        importance
        .head(12)
        .sort_values()
        .plot(
            kind="barh",
            ax=axis,
        )
    )

    axis.set_title(
        "Current-State Classification — "
        "Global SHAP Importance"
    )

    axis.set_xlabel(
        "mean(|SHAP value|)"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "11_shap_global_importance.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Low-class SHAP summary
    # --------------------------------------------------------------
    low_index = class_names.index(
        "Low"
    )

    if len(shap_per_class) > low_index:

        try:
            plt.figure(
                figsize=(9, 6)
            )

            shap.summary_plot(
                shap_per_class[
                    low_index
                ],
                X_sample,
                feature_names=feature_columns,
                show=False,
                max_display=12,
            )

            plt.title(
                "SHAP Drivers of 'Low' "
                "Current-State Classification"
            )

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR
                / "12_shap_low_class_summary.png",
                dpi=150,
            )

            plt.close()

        except Exception as error:

            print(
                "\nLow-class SHAP summary "
                "plot skipped:"
            )

            print(
                type(error).__name__,
                str(error),
            )

    # --------------------------------------------------------------
    # Example local explanations
    # --------------------------------------------------------------
    examples = []

    for class_name in class_names:

        matches = sample[
            sample["Efficiency_Status"]
            == class_name
        ]

        if matches.empty:
            continue

        row_index = matches.index[0]

        sample_position = (
            sample.index.get_loc(
                row_index
            )
        )

        class_index = (
            class_names.index(
                class_name
            )
        )

        if len(shap_per_class) == 1:
            row_shap = (
                shap_per_class[0][
                    sample_position
                ]
            )
        else:
            row_shap = (
                shap_per_class[
                    class_index
                ][
                    sample_position
                ]
            )

        contribution = pd.Series(
            row_shap,
            index=feature_columns,
        )

        top_features = (
            contribution
            .abs()
            .sort_values(
                ascending=False
            )
            .head(5)
            .index
        )

        examples.append(
            {
                "true_class": class_name,
                "row_index": int(
                    row_index
                ),
                "interpretation": (
                    "Current-state benchmark "
                    "classification explanation"
                ),
                "top_contributing_features": {
                    feature: float(
                        contribution[
                            feature
                        ]
                    )
                    for feature
                    in top_features
                },
            }
        )

    (
        OUTPUT_DIR
        / "shap_example_explanations.json"
    ).write_text(
        json.dumps(
            examples,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nExplainability analysis complete."
    )


if __name__ == "__main__":
    main()