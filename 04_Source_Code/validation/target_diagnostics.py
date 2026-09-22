from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from config import (
    DIAGNOSTICS_OUTPUT_DIR,
    VALIDATION_OUTPUT_DIR,
    RAW_DATA_PATH,
    ensure_output_directories,
)

from evaluation_utils import evaluate_classifier


TARGET = "Efficiency_Status"

RULE_FEATURES = [
    "Error_Rate_%",
    "Production_Speed_units_per_hr",
]

CLASS_LABELS = ["High", "Low", "Medium"]


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create a chronological timestamp from the raw date/time columns."""
    data = df.copy()

    data["Datetime"] = pd.to_datetime(
        data["Date"].astype(str) + " " + data["Timestamp"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="raise",
    )

    return data.sort_values("Datetime").reset_index(drop=True)


def business_rule_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Transparent approximation of the operational rule visible in the data.

    Low:
        Error rate > 5%, OR
        production speed < 200 units/hour

    High:
        Error rate <= 2% AND
        production speed >= 400 units/hour

    Medium:
        Remaining records
    """
    error = df["Error_Rate_%"].to_numpy()
    speed = df["Production_Speed_units_per_hr"].to_numpy()

    predictions = np.full(len(df), "Medium", dtype=object)

    low_mask = (error > 5) | (speed < 200)

    high_mask = (
        (error <= 2)
        & (speed >= 400)
        & (~low_mask)
    )

    predictions[low_mask] = "Low"
    predictions[high_mask] = "High"

    return predictions


def build_rule_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create the descriptive target-rule contingency table."""

    data = df.copy()

    data["Error_Band"] = np.select(
        [
            data["Error_Rate_%"] <= 2,
            data["Error_Rate_%"] <= 5,
        ],
        [
            "<=2%",
            "2-5%",
        ],
        default=">5%",
    )

    data["Speed_Band"] = np.select(
        [
            data["Production_Speed_units_per_hr"] < 200,
            data["Production_Speed_units_per_hr"] < 400,
        ],
        [
            "<200",
            "200-400",
        ],
        default=">=400",
    )

    table = pd.crosstab(
        [data["Error_Band"], data["Speed_Band"]],
        data[TARGET],
    )

    for label in CLASS_LABELS:
        if label not in table.columns:
            table[label] = 0

    return table[CLASS_LABELS]


def save_evaluation(
    prefix: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict:
    """Evaluate and persist one baseline."""

    summary, class_metrics, confusion = evaluate_classifier(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
    )

    class_metrics.to_csv(
        VALIDATION_OUTPUT_DIR / f"{prefix}_class_metrics.csv"
    )

    confusion.to_csv(
        VALIDATION_OUTPUT_DIR / f"{prefix}_confusion_matrix.csv"
    )

    return {
        key: float(value)
        for key, value in summary.items()
    }


def main() -> None:
    ensure_output_directories()

    df = pd.read_csv(RAW_DATA_PATH)
    df = build_datetime(df)

    print("=" * 78)
    print("TARGET CONSTRUCTION & BASELINE DIAGNOSTIC")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------
    split_idx = int(len(df) * 0.80)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nTotal records: {len(df):,}")
    print(f"Training records: {len(train_df):,}")
    print(f"Holdout records: {len(test_df):,}")

    print("\nHoldout target distribution:")
    print(
        test_df[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        + "%"
    )

    # ------------------------------------------------------------------
    # 1. Majority-class baseline
    # ------------------------------------------------------------------
    majority_class = train_df[TARGET].mode()[0]

    majority_pred = np.full(
        len(test_df),
        majority_class,
        dtype=object,
    )

    majority_metrics = save_evaluation(
        "majority_baseline",
        test_df[TARGET],
        majority_pred,
    )

    # ------------------------------------------------------------------
    # 2. Transparent business-rule baseline
    # ------------------------------------------------------------------
    rule_full_pred = business_rule_predict(df)

    rule_full_metrics = save_evaluation(
        "business_rule_full_dataset",
        df[TARGET],
        rule_full_pred,
    )

    rule_holdout_pred = business_rule_predict(test_df)

    rule_holdout_metrics = save_evaluation(
        "business_rule_holdout",
        test_df[TARGET],
        rule_holdout_pred,
    )

    # ------------------------------------------------------------------
    # 3. Shallow decision tree using only the two rule-linked variables
    # ------------------------------------------------------------------
    X_train = train_df[RULE_FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[RULE_FEATURES]
    y_test = test_df[TARGET]

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    tree_pred = tree.predict(X_test)

    tree_metrics = save_evaluation(
        "two_feature_tree_holdout",
        y_test,
        tree_pred,
    )

    tree_rules = export_text(
        tree,
        feature_names=RULE_FEATURES,
    )

    tree_path = (
        DIAGNOSTICS_OUTPUT_DIR
        / "two_feature_decision_tree_rules.txt"
    )

    tree_path.write_text(
        tree_rules,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 4. Descriptive rule table
    # ------------------------------------------------------------------
    rule_table = build_rule_table(df)

    rule_table_path = (
        DIAGNOSTICS_OUTPUT_DIR
        / "target_rule_contingency_table.csv"
    )

    rule_table.to_csv(rule_table_path)

    # ------------------------------------------------------------------
    # 5. Agreement / disagreement analysis
    # ------------------------------------------------------------------
    rule_errors = df.loc[
        df[TARGET].to_numpy() != rule_full_pred,
        [
            "Machine_ID",
            "Datetime",
            "Error_Rate_%",
            "Production_Speed_units_per_hr",
            TARGET,
        ],
    ].copy()

    rule_errors["Rule_Prediction"] = rule_full_pred[
        df[TARGET].to_numpy() != rule_full_pred
    ]

    rule_errors.to_csv(
        DIAGNOSTICS_OUTPUT_DIR
        / "business_rule_disagreements.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 6. Summary artifact
    # ------------------------------------------------------------------
    summary = {
        "dataset_rows": int(len(df)),
        "chronological_split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_fraction": 0.80,
            "test_fraction": 0.20,
        },
        "rule_features": RULE_FEATURES,
        "majority_class": str(majority_class),
        "majority_baseline": majority_metrics,
        "business_rule_full_dataset": rule_full_metrics,
        "business_rule_holdout": rule_holdout_metrics,
        "two_feature_decision_tree_holdout": tree_metrics,
        "business_rule_disagreements": int(len(rule_errors)),
    }

    summary_path = (
        DIAGNOSTICS_OUTPUT_DIR
        / "target_diagnostics_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("MAJORITY BASELINE")
    print("-" * 78)
    print(f"Predicted class:      {majority_class}")
    print(
        f"Accuracy:             "
        f"{majority_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{majority_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1:             "
        f"{majority_metrics['macro_f1']:.4f}"
    )

    print("\n" + "-" * 78)
    print("TRANSPARENT BUSINESS RULE — HOLDOUT")
    print("-" * 78)
    print(
        f"Accuracy:             "
        f"{rule_holdout_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{rule_holdout_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1:             "
        f"{rule_holdout_metrics['macro_f1']:.4f}"
    )

    print("\n" + "-" * 78)
    print("TWO-FEATURE DECISION TREE — HOLDOUT")
    print("-" * 78)
    print(
        f"Accuracy:             "
        f"{tree_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{tree_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1:             "
        f"{tree_metrics['macro_f1']:.4f}"
    )

    print("\nDecision tree rules:")
    print(tree_rules)

    print("-" * 78)
    print(
        "Business-rule disagreements across full dataset:",
        f"{len(rule_errors):,}",
    )

    print("\nArtifacts:")
    print(f"  {summary_path}")
    print(f"  {rule_table_path}")
    print(f"  {tree_path}")

    print("\nTarget diagnostic complete.")


if __name__ == "__main__":
    main()