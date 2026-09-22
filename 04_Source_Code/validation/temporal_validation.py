from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from config import (
    FEATURE_DATA_PATH,
    METRICS_OUTPUT_DIR,
    VALIDATION_OUTPUT_DIR,
    ensure_output_directories,
)

from evaluation_utils import evaluate_classifier


TARGET = "Efficiency_Status"

CLASS_NAMES = ["High", "Low", "Medium"]

LABEL_MAP = {
    "High": 0,
    "Low": 1,
    "Medium": 2,
}

INVERSE_LABEL_MAP = {
    value: key
    for key, value in LABEL_MAP.items()
}


BASE_NUMERIC_FEATURES = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Sensor_Stability_Score",
    "Energy_Efficiency_Ratio",
    "Error_to_Output_Ratio",
    "Quality_Adjusted_Error",
    "Network_Reliability_Score",
    "Hour",
    "Is_Weekend",
]


DIRECT_RULE_FEATURES = [
    "Error_Rate_%",
    "Production_Speed_units_per_hr",
]


RULE_LINKED_FEATURES = [
    "Error_Rate_%",
    "Production_Speed_units_per_hr",
    "Energy_Efficiency_Ratio",
    "Error_to_Output_Ratio",
    "Quality_Adjusted_Error",
]


SENSOR_NETWORK_FEATURES = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Sensor_Stability_Score",
    "Network_Reliability_Score",
    "Hour",
    "Is_Weekend",
]


def prepare_data() -> tuple[pd.DataFrame, list[str]]:
    """Load, sort and encode the engineered dataset."""

    df = pd.read_parquet(FEATURE_DATA_PATH)

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df = (
        df
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    df = pd.get_dummies(
        df,
        columns=["Operation_Mode"],
        prefix="Mode",
        dtype=int,
    )

    mode_columns = [
        column
        for column in df.columns
        if column.startswith("Mode_")
    ]

    return df, mode_columns


def encode_target(series: pd.Series) -> np.ndarray:
    return (
        series
        .map(LABEL_MAP)
        .astype(int)
        .to_numpy()
    )


def decode_target(values: np.ndarray) -> np.ndarray:
    return np.array(
        [INVERSE_LABEL_MAP[int(value)] for value in values],
        dtype=object,
    )


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights calculated only from the training fold."""

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    total = len(y)
    n_classes = len(classes)

    return {
        int(cls): float(total / (n_classes * count))
        for cls, count in zip(classes, counts)
    }


def sample_weights(y: np.ndarray) -> np.ndarray:
    weights = class_weights(y)

    return np.array(
        [weights[int(value)] for value in y],
        dtype=float,
    )


def build_model(
    model_name: str,
    y_train: np.ndarray,
):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def fit_predict(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
) -> np.ndarray:
    """
    Fit one validation model.

    Scaling is learned from the training fold only.
    """

    model = build_model(
        model_name,
        y_train,
    )

    if model_name == "Logistic Regression":
        scaler = StandardScaler()

        X_train_model = scaler.fit_transform(
            X_train
        )

        X_valid_model = scaler.transform(
            X_valid
        )

        model.fit(
            X_train_model,
            y_train,
        )

        return model.predict(
            X_valid_model
        )

    if model_name == "XGBoost":
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights(y_train),
        )

        return model.predict(
            X_valid
        )

    model.fit(
        X_train,
        y_train,
    )

    return model.predict(
        X_valid
    )


def evaluate_predictions(
    y_true_labels: pd.Series,
    pred_numeric: np.ndarray,
) -> dict:
    pred_labels = decode_target(
        pred_numeric
    )

    summary, class_metrics, _ = evaluate_classifier(
        y_true_labels,
        pred_labels,
        labels=CLASS_NAMES,
    )

    high_recall = float(
        class_metrics.loc["High", "recall"]
    )

    high_precision = float(
        class_metrics.loc["High", "precision"]
    )

    return {
        **{
            key: float(value)
            for key, value in summary.items()
        },
        "high_precision": high_precision,
        "high_recall": high_recall,
    }


def temporal_model_validation(
    development_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Expanding-window validation on the development period.

    This replaces shuffled StratifiedKFold.
    """

    splitter = TimeSeriesSplit(
        n_splits=4
    )

    models = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]

    results = []

    for model_name in models:

        for fold, (
            train_idx,
            valid_idx,
        ) in enumerate(
            splitter.split(development_df),
            start=1,
        ):

            train_fold = development_df.iloc[
                train_idx
            ]

            valid_fold = development_df.iloc[
                valid_idx
            ]

            X_train = train_fold[
                feature_columns
            ]

            X_valid = valid_fold[
                feature_columns
            ]

            y_train = encode_target(
                train_fold[TARGET]
            )

            predictions = fit_predict(
                model_name,
                X_train,
                y_train,
                X_valid,
            )

            metrics = evaluate_predictions(
                valid_fold[TARGET],
                predictions,
            )

            results.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_rows": len(train_fold),
                    "validation_rows": len(valid_fold),
                    "validation_start": str(
                        valid_fold["Datetime"].min()
                    ),
                    "validation_end": str(
                        valid_fold["Datetime"].max()
                    ),
                    **metrics,
                }
            )

            print(
                f"{model_name:<22} "
                f"fold={fold} "
                f"macro_f1="
                f"{metrics['macro_f1']:.4f} "
                f"balanced_acc="
                f"{metrics['balanced_accuracy']:.4f} "
                f"high_recall="
                f"{metrics['high_recall']:.4f}"
            )

    return pd.DataFrame(
        results
    )


def holdout_model_validation(
    development_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Evaluate all three models on the untouched final holdout."""

    rows = []

    for model_name in [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]:

        X_train = development_df[
            feature_columns
        ]

        X_test = holdout_df[
            feature_columns
        ]

        y_train = encode_target(
            development_df[TARGET]
        )

        predictions = fit_predict(
            model_name,
            X_train,
            y_train,
            X_test,
        )

        metrics = evaluate_predictions(
            holdout_df[TARGET],
            predictions,
        )

        rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def random_forest_ablation(
    development_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compare feature groups using the same Random Forest architecture.

    This isolates how much performance depends on rule-linked features.
    """

    rows = []

    for experiment_name, features in feature_sets.items():

        X_train = development_df[
            features
        ]

        X_test = holdout_df[
            features
        ]

        y_train = encode_target(
            development_df[TARGET]
        )

        predictions = fit_predict(
            "Random Forest",
            X_train,
            y_train,
            X_test,
        )

        metrics = evaluate_predictions(
            holdout_df[TARGET],
            predictions,
        )

        rows.append(
            {
                "experiment": experiment_name,
                "feature_count": len(features),
                **metrics,
            }
        )

        print(
            f"{experiment_name:<30} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_acc="
            f"{metrics['balanced_accuracy']:.4f} "
            f"high_recall="
            f"{metrics['high_recall']:.4f}"
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    ensure_output_directories()

    df, mode_columns = prepare_data()

    all_features = (
        BASE_NUMERIC_FEATURES
        + mode_columns
    )

    no_direct_rule = [
        feature
        for feature in all_features
        if feature not in DIRECT_RULE_FEATURES
    ]

    no_rule_linked = [
        feature
        for feature in all_features
        if feature not in RULE_LINKED_FEATURES
    ]

    sensor_network_only = (
        SENSOR_NETWORK_FEATURES
        + mode_columns
    )

    feature_sets = {
        "all_features": all_features,
        "no_direct_rule_features": no_direct_rule,
        "no_rule_linked_features": no_rule_linked,
        "sensor_network_only": sensor_network_only,
    }

    # --------------------------------------------------------------
    # Final 20% remains completely untouched during CV
    # --------------------------------------------------------------
    split_index = int(
        len(df) * 0.80
    )

    development_df = (
        df.iloc[:split_index]
        .copy()
    )

    holdout_df = (
        df.iloc[split_index:]
        .copy()
    )

    print("=" * 80)
    print("TIME-AWARE MODEL VALIDATION")
    print("=" * 80)

    print(
        f"Development rows: {len(development_df):,}"
    )

    print(
        f"Final holdout rows: {len(holdout_df):,}"
    )

    # --------------------------------------------------------------
    # Expanding-window CV — all original features
    # --------------------------------------------------------------
    print("\nExpanding-window validation:")

    fold_results = temporal_model_validation(
        development_df,
        all_features,
    )

    fold_results.to_csv(
        VALIDATION_OUTPUT_DIR
        / "temporal_fold_results.csv",
        index=False,
    )

    temporal_summary = (
        fold_results
        .groupby("model")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            high_recall_mean=(
                "high_recall",
                "mean",
            ),
            high_recall_std=(
                "high_recall",
                "std",
            ),
        )
        .reset_index()
    )

    temporal_summary.to_csv(
        METRICS_OUTPUT_DIR
        / "temporal_model_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Untouched final holdout
    # --------------------------------------------------------------
    print("\nFinal holdout evaluation:")

    holdout_results = holdout_model_validation(
        development_df,
        holdout_df,
        all_features,
    )

    print(
        holdout_results[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "high_precision",
                "high_recall",
            ]
        ].to_string(
            index=False
        )
    )

    holdout_results.to_csv(
        METRICS_OUTPUT_DIR
        / "holdout_model_results.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Feature ablation
    # --------------------------------------------------------------
    print("\nFeature ablation:")

    ablation_results = random_forest_ablation(
        development_df,
        holdout_df,
        feature_sets,
    )

    ablation_results.to_csv(
        METRICS_OUTPUT_DIR
        / "feature_ablation_results.csv",
        index=False,
    )

    metadata = {
        "validation_method": (
            "4-fold expanding-window validation "
            "within first 80% of chronological data, "
            "followed by untouched final 20% holdout"
        ),
        "development_rows": int(
            len(development_df)
        ),
        "holdout_rows": int(
            len(holdout_df)
        ),
        "feature_sets": feature_sets,
        "primary_metric": "macro_f1",
        "secondary_metrics": [
            "balanced_accuracy",
            "high_recall",
            "weighted_f1",
        ],
    }

    (
        VALIDATION_OUTPUT_DIR
        / "validation_methodology.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(
        "\nSaved temporal folds, final holdout metrics, "
        "and feature-ablation results."
    )


if __name__ == "__main__":
    main()