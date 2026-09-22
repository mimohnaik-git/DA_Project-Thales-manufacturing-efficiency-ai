"""
03_modeling.py

Final current-state efficiency classification benchmark.

Important methodological distinction:
- Efficiency_Status is almost deterministically reconstructable from
  Error_Rate_% and Production_Speed_units_per_hr.
- The ML models in this file are therefore benchmark models for
  current-state rule reconstruction, not validated future-risk models.
- Future prediction was rejected after temporal diagnostics showed
  effectively zero temporal dependence in the dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


SOURCE_DIR = Path(__file__).resolve().parent

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


from config import (
    FEATURE_DATA_PATH,
    FIGURES_DIR,
    METRICS_OUTPUT_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    ensure_output_directories,
)

from validation.evaluation_utils import evaluate_classifier


TARGET = "Efficiency_Status"

CATEGORICAL_FEATURES = [
    "Operation_Mode",
]

NUMERIC_FEATURES = [
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


def business_rule_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Transparent rule discovered during target diagnostics.

    Low:
      Error_Rate_% > 5
      OR Production_Speed_units_per_hr < 200

    High:
      Error_Rate_% <= 2
      AND Production_Speed_units_per_hr >= 400

    Medium:
      Remaining observations
    """

    error_rate = df["Error_Rate_%"].to_numpy()
    production_speed = (
        df["Production_Speed_units_per_hr"]
        .to_numpy()
    )

    prediction = np.full(
        len(df),
        "Medium",
        dtype=object,
    )

    low_mask = (
        (error_rate > 5)
        | (production_speed < 200)
    )

    high_mask = (
        (error_rate <= 2)
        & (production_speed >= 400)
        & (~low_mask)
    )

    prediction[low_mask] = "Low"
    prediction[high_mask] = "High"

    return prediction


def balanced_sample_weights(
    y: np.ndarray,
) -> np.ndarray:

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    total = len(y)
    number_classes = len(classes)

    weights = {
        int(cls): (
            total
            / (
                number_classes
                * count
            )
        )
        for cls, count
        in zip(classes, counts)
    }

    return np.array(
        [
            weights[int(value)]
            for value in y
        ],
        dtype=float,
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Multiclass calibration error based on
    maximum predicted probability.
    """

    confidence = probabilities.max(
        axis=1
    )

    prediction = probabilities.argmax(
        axis=1
    )

    correct = (
        prediction == y_true
    ).astype(float)

    boundaries = np.linspace(
        0,
        1,
        bins + 1,
    )

    ece = 0.0

    for lower, upper in zip(
        boundaries[:-1],
        boundaries[1:],
    ):

        if upper == 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if not np.any(mask):
            continue

        bin_accuracy = (
            correct[mask].mean()
        )

        bin_confidence = (
            confidence[mask].mean()
        )

        ece += (
            np.abs(
                bin_accuracy
                - bin_confidence
            )
            * mask.mean()
        )

    return float(ece)


def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Probability-quality metrics.

    These prevent raw model probability from
    being presented as inherently trustworthy
    prediction confidence.
    """

    n_classes = len(class_names)

    one_hot = np.eye(
        n_classes
    )[y_true]

    macro_auc = roc_auc_score(
        one_hot,
        probabilities,
        average="macro",
        multi_class="ovr",
    )

    macro_average_precision = (
        average_precision_score(
            one_hot,
            probabilities,
            average="macro",
        )
    )

    high_index = class_names.index(
        "High"
    )

    high_average_precision = (
        average_precision_score(
            one_hot[:, high_index],
            probabilities[:, high_index],
        )
    )

    multiclass_brier = np.mean(
        np.sum(
            (
                probabilities
                - one_hot
            )
            ** 2,
            axis=1,
        )
    )

    return {
        "roc_auc_macro_ovr": float(
            macro_auc
        ),
        "average_precision_macro": float(
            macro_average_precision
        ),
        "high_class_average_precision": float(
            high_average_precision
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=np.arange(
                    n_classes
                ),
            )
        ),
        "multiclass_brier_score": float(
            multiclass_brier
        ),
        "expected_calibration_error": (
            expected_calibration_error(
                y_true,
                probabilities,
            )
        ),
    }


def evaluate_label_predictions(
    y_true_labels,
    y_pred_labels,
    class_names,
):
    summary, class_metrics, confusion = (
        evaluate_classifier(
            y_true_labels,
            y_pred_labels,
            labels=class_names,
        )
    )

    return (
        {
            key: float(value)
            for key, value
            in summary.items()
        },
        class_metrics,
        confusion,
    )


def evaluate_model(
    model_name: str,
    y_true_numeric: np.ndarray,
    y_pred_numeric: np.ndarray,
    probabilities: np.ndarray,
    label_encoder: LabelEncoder,
):
    y_true_labels = (
        label_encoder.inverse_transform(
            y_true_numeric
        )
    )

    y_pred_labels = (
        label_encoder.inverse_transform(
            y_pred_numeric
        )
    )

    class_names = list(
        label_encoder.classes_
    )

    (
        summary,
        class_metrics,
        confusion,
    ) = evaluate_label_predictions(
        y_true_labels,
        y_pred_labels,
        class_names,
    )

    summary.update(
        probability_metrics(
            y_true_numeric,
            probabilities,
            class_names,
        )
    )

    class_metrics.to_csv(
        METRICS_OUTPUT_DIR
        / (
            model_name.lower()
            .replace(" ", "_")
            + "_class_metrics.csv"
        )
    )

    confusion.to_csv(
        METRICS_OUTPUT_DIR
        / (
            model_name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.csv"
        )
    )

    return (
        summary,
        class_metrics,
        confusion,
    )


def main() -> None:
    ensure_output_directories()

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

    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_FEATURES,
        prefix="Mode",
        dtype=int,
    )

    mode_columns = [
        column
        for column
        in df_encoded.columns
        if column.startswith("Mode_")
    ]

    feature_columns = (
        NUMERIC_FEATURES
        + mode_columns
    )

    label_encoder = LabelEncoder()

    df_encoded["target"] = pd.Series(
        np.asarray(
            label_encoder.fit_transform(
                df_encoded[TARGET]
            ),
            dtype=np.int64,
        ),
        index=df_encoded.index,
    )

    class_names = list(
        label_encoder.classes_
    )

    split_index = int(
        len(df_encoded) * 0.80
    )

    train_df = (
        df_encoded.iloc[
            :split_index
        ].copy()
    )

    test_df = (
        df_encoded.iloc[
            split_index:
        ].copy()
    )

    X_train = train_df[
        feature_columns
    ]

    X_test = test_df[
        feature_columns
    ]

    y_train = train_df[
        "target"
    ].to_numpy()

    y_test = test_df[
        "target"
    ].to_numpy()

    y_test_labels = test_df[
        TARGET
    ].to_numpy()

    print("=" * 78)
    print(
        "FINAL CURRENT-STATE CLASSIFICATION BENCHMARK"
    )
    print("=" * 78)

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Holdout rows:  {len(test_df):,}"
    )

    print(
        f"Classes:       {class_names}"
    )

    # --------------------------------------------------------------
    # Baseline 1 — majority class
    # --------------------------------------------------------------
    majority_class = (
        train_df[TARGET]
        .mode()[0]
    )

    majority_prediction = (
        np.full(
            len(test_df),
            majority_class,
            dtype=object,
        )
    )

    (
        majority_metrics,
        _,
        majority_confusion,
    ) = evaluate_label_predictions(
        y_test_labels,
        majority_prediction,
        class_names,
    )

    # --------------------------------------------------------------
    # Baseline 2 — transparent business rule
    # --------------------------------------------------------------
    rule_prediction = (
        business_rule_predict(
            test_df
        )
    )

    (
        rule_metrics,
        _,
        rule_confusion,
    ) = evaluate_label_predictions(
        y_test_labels,
        rule_prediction,
        class_names,
    )

    # --------------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[
        NUMERIC_FEATURES
    ] = scaler.fit_transform(
        X_train[
            NUMERIC_FEATURES
        ]
    )

    X_test_scaled[
        NUMERIC_FEATURES
    ] = scaler.transform(
        X_test[
            NUMERIC_FEATURES
        ]
    )

    logistic_model = (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    )

    logistic_model.fit(
        X_train_scaled,
        y_train,
    )

    logistic_prediction = (
        logistic_model.predict(
            X_test_scaled
        )
    )

    logistic_probability = (
        logistic_model.predict_proba(
            X_test_scaled
        )
    )

    # --------------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------------
    random_forest_model = (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    random_forest_model.fit(
        X_train,
        y_train,
    )

    random_forest_prediction = (
        random_forest_model.predict(
            X_test
        )
    )

    random_forest_probability = (
        random_forest_model.predict_proba(
            X_test
        )
    )

    # --------------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------------
    xgboost_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    xgboost_model.fit(
        X_train,
        y_train,
        sample_weight=(
            balanced_sample_weights(
                y_train
            )
        ),
    )

    xgboost_prediction = (
        xgboost_model.predict(
            X_test
        )
    )

    xgboost_probability = (
        xgboost_model.predict_proba(
            X_test
        )
    )

    # --------------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------------
    model_outputs = {
        "Logistic Regression": (
            logistic_prediction,
            logistic_probability,
        ),
        "Random Forest": (
            random_forest_prediction,
            random_forest_probability,
        ),
        "XGBoost": (
            xgboost_prediction,
            xgboost_probability,
        ),
    }

    model_results = {}
    model_confusions = {}

    for (
        model_name,
        (
            prediction,
            probability,
        ),
    ) in model_outputs.items():

        (
            metrics,
            _,
            confusion,
        ) = evaluate_model(
            model_name,
            y_test,
            prediction,
            probability,
            label_encoder,
        )

        model_results[
            model_name
        ] = metrics

        model_confusions[
            model_name
        ] = confusion

    # --------------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------------
    comparison_rows = [
        {
            "method": "Majority Baseline",
            **majority_metrics,
        },
        {
            "method": "Transparent Business Rule",
            **rule_metrics,
        },
    ]

    for model_name, metrics in (
        model_results.items()
    ):
        comparison_rows.append(
            {
                "method": model_name,
                **metrics,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        METRICS_OUTPUT_DIR
        / "final_model_comparison.csv",
        index=False,
    )

    print("\nFinal comparison:")

    display_columns = [
        "method",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Benchmark model selection
    # --------------------------------------------------------------
    benchmark_model_name = max(
        model_results,
        key=lambda name: (
            model_results[name][
                "macro_f1"
            ]
        ),
    )

    print(
        "\nBest ML benchmark by Macro F1:",
        benchmark_model_name,
    )

    print(
        "Primary operational decision method:",
        "Transparent Business Rule",
    )

    # --------------------------------------------------------------
    # Persist models
    # --------------------------------------------------------------
    joblib.dump(
        logistic_model,
        MODEL_DIR
        / "logistic_regression_model.joblib",
    )

    joblib.dump(
        random_forest_model,
        MODEL_DIR
        / "random_forest_model.joblib",
    )

    xgboost_model.save_model(
        MODEL_DIR
        / "xgboost_model.json"
    )

    joblib.dump(
        scaler,
        MODEL_DIR
        / "scaler.joblib",
    )

    joblib.dump(
        label_encoder,
        MODEL_DIR
        / "label_encoder.joblib",
    )

    # --------------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------------
    rf_importance = pd.Series(
        random_forest_model.feature_importances_,
        index=feature_columns,
    ).sort_values(
        ascending=False
    )

    xgb_importance = pd.Series(
        xgboost_model.feature_importances_,
        index=feature_columns,
    ).sort_values(
        ascending=False
    )

    rf_importance.to_csv(
        OUTPUT_DIR
        / "rf_feature_importance.csv"
    )

    xgb_importance.to_csv(
        OUTPUT_DIR
        / "xgb_feature_importance.csv"
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------
    metadata = {
        "task": (
            "Current-state Efficiency_Status "
            "classification benchmark"
        ),
        "task_interpretation": (
            "Rule reconstruction / automation; "
            "not validated future-risk prediction"
        ),
        "future_prediction_supported": False,
        "future_prediction_limitation": (
            "Temporal diagnostics found "
            "approximately zero status dependence "
            "and near-zero within-machine feature "
            "autocorrelation."
        ),
        "feature_columns": (
            feature_columns
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "mode_columns": mode_columns,
        "class_names": class_names,
        "split_index": split_index,
        "train_size": int(
            len(train_df)
        ),
        "test_size": int(
            len(test_df)
        ),
        "primary_metric": (
            "macro_f1"
        ),
        "primary_operational_method": (
            "Transparent Business Rule"
        ),
        "best_model": (
            benchmark_model_name
        ),
        "benchmark_model": (
            benchmark_model_name
        ),
        "rule_definition": {
            "low": (
                "Error_Rate_% > 5 OR "
                "Production_Speed_units_per_hr < 200"
            ),
            "high": (
                "Error_Rate_% <= 2 AND "
                "Production_Speed_units_per_hr >= 400"
            ),
            "medium": "Otherwise",
        },
    }

    (
        MODEL_DIR
        / "model_meta.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_results = {
        "methodology": {
            "split": (
                "Chronological 80/20 holdout"
            ),
            "primary_metric": (
                "Macro F1"
            ),
            "imbalance_metrics": [
                "Balanced Accuracy",
                "Macro F1",
                "Weighted F1",
                "Per-class Precision",
                "Per-class Recall",
                "Average Precision",
            ],
            "probability_quality": [
                "Log Loss",
                "Multiclass Brier Score",
                "Expected Calibration Error",
            ],
        },
        "baselines": {
            "Majority Baseline": (
                majority_metrics
            ),
            "Transparent Business Rule": (
                rule_metrics
            ),
        },
        "models": model_results,
        "selection": {
            "primary_operational_method": (
                "Transparent Business Rule"
            ),
            "ml_benchmark_model": (
                benchmark_model_name
            ),
        },
    }

    (
        OUTPUT_DIR
        / "model_results.json"
    ).write_text(
        json.dumps(
            final_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Figure 08 — confusion matrices
    # --------------------------------------------------------------
    confusion_items = [
        (
            "Business Rule",
            rule_confusion,
        ),
        (
            "Logistic Regression",
            model_confusions[
                "Logistic Regression"
            ],
        ),
        (
            "Random Forest",
            model_confusions[
                "Random Forest"
            ],
        ),
        (
            "XGBoost",
            model_confusions[
                "XGBoost"
            ],
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 9),
    )

    for (
        axis,
        (
            title,
            matrix,
        ),
    ) in zip(
        axes.flat,
        confusion_items,
    ):

        image = axis.imshow(
            matrix.values
        )

        axis.set_title(
            title
        )

        axis.set_xticks(
            range(
                len(class_names)
            ),
            class_names,
        )

        axis.set_yticks(
            range(
                len(class_names)
            ),
            class_names,
        )

        axis.set_xlabel(
            "Predicted"
        )

        axis.set_ylabel(
            "Actual"
        )

        for row in range(
            matrix.shape[0]
        ):
            for column in range(
                matrix.shape[1]
            ):
                axis.text(
                    column,
                    row,
                    int(
                        matrix.iloc[
                            row,
                            column,
                        ]
                    ),
                    ha="center",
                    va="center",
                )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "08_confusion_matrices.png",
        dpi=150,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------------
    # Figure 09 — evaluation metrics
    # --------------------------------------------------------------
    comparison_plot = (
        comparison[
            [
                "method",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ]
        ]
        .set_index(
            "method"
        )
    )

    axis = comparison_plot.plot(
        kind="bar",
        figsize=(12, 6),
    )

    axis.set_ylim(
        0,
        1.05,
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_title(
        "Current-State Classification: "
        "Baselines vs ML Models"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "09_model_comparison.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------------
    # Figure 10 — feature importance
    # --------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    (
        rf_importance
        .head(12)
        .sort_values()
        .plot(
            kind="barh",
            ax=axes[0],
        )
    )

    axes[0].set_title(
        "Random Forest Feature Importance"
    )

    (
        xgb_importance
        .head(12)
        .sort_values()
        .plot(
            kind="barh",
            ax=axes[1],
        )
    )

    axes[1].set_title(
        "XGBoost Feature Importance"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "10_feature_importance.png",
        dpi=150,
    )

    plt.close(
        fig
    )

    print(
        "\nModel artifacts and "
        "evaluation outputs saved."
    )


if __name__ == "__main__":
    main()