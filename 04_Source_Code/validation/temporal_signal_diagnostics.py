from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import mutual_info_score


SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from config import (
    DIAGNOSTICS_OUTPUT_DIR,
    FEATURE_DATA_PATH,
    ensure_output_directories,
)


TARGET = "Efficiency_Status"

CLASSES = [
    "High",
    "Low",
    "Medium",
]

HORIZONS = [1, 3, 5, 10]

TEMPORAL_FEATURES = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
]


def prepare_data() -> pd.DataFrame:
    df = pd.read_parquet(
        FEATURE_DATA_PATH
    )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"]
    )

    return (
        df.sort_values(
            ["Machine_ID", "Datetime"]
        )
        .reset_index(drop=True)
    )


def expected_independent_agreement(
    current: pd.Series,
    future: pd.Series,
) -> float:
    """
    Probability that current and future labels match
    if they are independent but retain their observed
    marginal class distributions.
    """

    current_prob = (
        current.value_counts(normalize=True)
    )

    future_prob = (
        future.value_counts(normalize=True)
    )

    return float(
        sum(
            current_prob.get(label, 0.0)
            * future_prob.get(label, 0.0)
            for label in CLASSES
        )
    )


def horizon_analysis(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[dict, pd.DataFrame]:
    data = df.copy()

    grouped = data.groupby(
        "Machine_ID",
        sort=False,
    )

    data["Future_Status"] = grouped[
        TARGET
    ].shift(-horizon)

    valid = data[
        data["Future_Status"].notna()
    ].copy()

    current = valid[TARGET]
    future = valid["Future_Status"]

    observed_agreement = float(
        (current == future).mean()
    )

    expected_agreement = (
        expected_independent_agreement(
            current,
            future,
        )
    )

    kappa = float(
        cohen_kappa_score(
            current,
            future,
            labels=CLASSES,
        )
    )

    mutual_information = float(
        mutual_info_score(
            current,
            future,
        )
    )

    transition_matrix = pd.crosstab(
        current,
        future,
        normalize="index",
    )

    transition_matrix = (
        transition_matrix
        .reindex(
            index=CLASSES,
            columns=CLASSES,
            fill_value=0,
        )
    )

    future_low_by_current = {
        label: float(
            transition_matrix.to_numpy(dtype=float)[
                CLASSES.index(label),
                CLASSES.index("Low"),
            ]
        )
        for label in CLASSES
    }

    result = {
        "horizon": horizon,
        "valid_rows": int(
            len(valid)
        ),
        "observed_same_status_rate": (
            observed_agreement
        ),
        "expected_same_status_if_independent": (
            expected_agreement
        ),
        "agreement_above_chance": float(
            observed_agreement
            - expected_agreement
        ),
        "cohen_kappa": kappa,
        "mutual_information": (
            mutual_information
        ),
        "future_low_rate_by_current_status": (
            future_low_by_current
        ),
    }

    return (
        result,
        transition_matrix,
    )


def lag_one_feature_autocorrelation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate lag-1 autocorrelation separately
    within each machine, then summarize across machines.
    """

    rows = []

    for feature in TEMPORAL_FEATURES:

        machine_correlations = []

        for _, machine_df in df.groupby(
            "Machine_ID",
            sort=False,
        ):

            series = (
                machine_df[feature]
                .astype(float)
            )

            if (
                len(series) < 3
                or series.nunique() <= 1
            ):
                continue

            correlation = series.autocorr(
                lag=1
            )

            if pd.notna(correlation):
                machine_correlations.append(
                    float(correlation)
                )

        if machine_correlations:
            values = np.asarray(
                machine_correlations
            )

            rows.append(
                {
                    "feature": feature,
                    "machines_evaluated": (
                        len(values)
                    ),
                    "mean_lag1_autocorrelation": (
                        float(values.mean())
                    ),
                    "median_lag1_autocorrelation": (
                        float(np.median(values))
                    ),
                    "mean_absolute_lag1_autocorrelation": (
                        float(
                            np.abs(values).mean()
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    ensure_output_directories()

    df = prepare_data()

    print("=" * 80)
    print("TEMPORAL SIGNAL DIAGNOSTIC")
    print("=" * 80)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Machines: "
        f"{df['Machine_ID'].nunique():,}"
    )

    horizon_results = []

    for horizon in HORIZONS:

        (
            result,
            transition_matrix,
        ) = horizon_analysis(
            df,
            horizon,
        )

        horizon_results.append(
            result
        )

        transition_matrix.to_csv(
            DIAGNOSTICS_OUTPUT_DIR
            / (
                "transition_matrix_"
                f"h{horizon}.csv"
            )
        )

        print("\n" + "-" * 80)

        print(
            f"Horizon: {horizon} observation(s)"
        )

        print(
            "Observed same-status rate:       "
            f"{result['observed_same_status_rate']:.4f}"
        )

        print(
            "Expected agreement if independent:"
            f" {result['expected_same_status_if_independent']:.4f}"
        )

        print(
            "Agreement above chance:          "
            f"{result['agreement_above_chance']:.4f}"
        )

        print(
            "Cohen's kappa:                   "
            f"{result['cohen_kappa']:.4f}"
        )

        print(
            "Mutual information:              "
            f"{result['mutual_information']:.6f}"
        )

        print("\nFuture Low probability:")

        for (
            current_status,
            probability,
        ) in result[
            "future_low_rate_by_current_status"
        ].items():

            print(
                f"  Current {current_status:<6}"
                f" -> Future Low: "
                f"{probability:.4f}"
            )

    autocorrelation = (
        lag_one_feature_autocorrelation(
            df
        )
    )

    autocorrelation.to_csv(
        DIAGNOSTICS_OUTPUT_DIR
        / "feature_lag1_autocorrelation.csv",
        index=False,
    )

    print("\n" + "=" * 80)

    print(
        "LAG-1 FEATURE AUTOCORRELATION"
    )

    print("=" * 80)

    print(
        autocorrelation.to_string(
            index=False
        )
    )

    summary = {
        "interpretation": (
            "Temporal dependence should be considered "
            "meaningful only when observed agreement "
            "materially exceeds independent agreement, "
            "Cohen's kappa is above zero by a useful "
            "margin, and operational features exhibit "
            "non-trivial within-machine autocorrelation."
        ),
        "horizons": horizon_results,
    }

    output_path = (
        DIAGNOSTICS_OUTPUT_DIR
        / "temporal_signal_diagnostics.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()