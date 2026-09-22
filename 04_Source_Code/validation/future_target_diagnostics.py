from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from config import (
    DIAGNOSTICS_OUTPUT_DIR,
    FEATURE_DATA_PATH,
    ensure_output_directories,
)


TARGET = "Efficiency_Status"

HORIZONS = [1, 3, 5, 10]


def prepare_data() -> pd.DataFrame:
    """Load engineered records in machine/time order."""

    df = pd.read_parquet(FEATURE_DATA_PATH)

    df["Datetime"] = pd.to_datetime(
        df["Datetime"]
    )

    df = (
        df.sort_values(
            ["Machine_ID", "Datetime"]
        )
        .reset_index(drop=True)
    )

    return df


def calculate_sampling_interval(
    df: pd.DataFrame,
) -> dict:
    """Describe the observation frequency within machines."""

    gaps = (
        df.groupby("Machine_ID")["Datetime"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    gaps = gaps[
        gaps > 0
    ].dropna()

    return {
        "observations": int(len(gaps)),
        "median_minutes": float(gaps.median()),
        "mean_minutes": float(gaps.mean()),
        "p25_minutes": float(gaps.quantile(0.25)),
        "p75_minutes": float(gaps.quantile(0.75)),
        "minimum_minutes": float(gaps.min()),
        "maximum_minutes": float(gaps.max()),
    }


def analyze_horizon(
    df: pd.DataFrame,
    horizon: int,
) -> dict:
    """
    Measure future-state persistence and transitions.

    Future labels are created independently within each machine.
    """

    data = df.copy()

    grouped = data.groupby(
        "Machine_ID",
        sort=False,
    )

    data["Future_Status"] = grouped[
        TARGET
    ].shift(-horizon)

    data["Future_Datetime"] = grouped[
        "Datetime"
    ].shift(-horizon)

    valid = data[
        data["Future_Status"].notna()
    ].copy()

    valid["Horizon_Minutes"] = (
        valid["Future_Datetime"]
        - valid["Datetime"]
    ).dt.total_seconds() / 60

    valid["Same_Status"] = (
        valid[TARGET]
        == valid["Future_Status"]
    )

    # Operational degradation population:
    # machines not already Low at prediction time.
    at_risk = valid[
        valid[TARGET] != "Low"
    ].copy()

    at_risk["Transition_To_Low"] = (
        at_risk["Future_Status"]
        == "Low"
    )

    positive_count = int(
        at_risk["Transition_To_Low"].sum()
    )

    at_risk_count = int(
        len(at_risk)
    )

    transition_rate = (
        positive_count / at_risk_count
        if at_risk_count
        else 0.0
    )

    return {
        "horizon_observations": horizon,
        "valid_rows": int(len(valid)),
        "median_horizon_minutes": float(
            valid["Horizon_Minutes"].median()
        ),
        "mean_horizon_minutes": float(
            valid["Horizon_Minutes"].mean()
        ),
        "status_persistence_rate": float(
            valid["Same_Status"].mean()
        ),
        "non_low_starting_rows": at_risk_count,
        "transitions_to_low": positive_count,
        "transition_to_low_rate": float(
            transition_rate
        ),
        "future_distribution": {
            str(label): int(count)
            for label, count in (
                valid["Future_Status"]
                .value_counts()
                .items()
            )
        },
    }


def main() -> None:
    ensure_output_directories()

    df = prepare_data()

    print("=" * 78)
    print("FUTURE TARGET DIAGNOSTIC")
    print("=" * 78)

    print(f"Rows: {len(df):,}")
    print(
        f"Machines: "
        f"{df['Machine_ID'].nunique():,}"
    )

    sampling = calculate_sampling_interval(
        df
    )

    print("\nSampling interval:")
    print(
        f"Median: "
        f"{sampling['median_minutes']:.2f} minutes"
    )
    print(
        f"Mean:   "
        f"{sampling['mean_minutes']:.2f} minutes"
    )
    print(
        f"IQR:    "
        f"{sampling['p25_minutes']:.2f} - "
        f"{sampling['p75_minutes']:.2f} minutes"
    )

    results = []

    print("\nFuture-target candidates:")
    print("-" * 78)

    for horizon in HORIZONS:

        result = analyze_horizon(
            df,
            horizon,
        )

        results.append(result)

        print(
            f"h={horizon:<2} | "
            f"median horizon="
            f"{result['median_horizon_minutes']:>7.2f} min | "
            f"persistence="
            f"{result['status_persistence_rate']:.4f} | "
            f"Low transitions="
            f"{result['transitions_to_low']:>6,} | "
            f"transition rate="
            f"{result['transition_to_low_rate']:.4f}"
        )

    summary = {
        "dataset_rows": int(len(df)),
        "machines": int(
            df["Machine_ID"].nunique()
        ),
        "sampling_interval": sampling,
        "candidate_horizons": results,
        "recommended_target_definition": (
            "Among observations where current Efficiency_Status "
            "is not Low, predict whether that machine transitions "
            "to Low at a future horizon."
        ),
    }

    output_json = (
        DIAGNOSTICS_OUTPUT_DIR
        / "future_target_diagnostics.json"
    )

    output_json.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_csv = (
        DIAGNOSTICS_OUTPUT_DIR
        / "future_horizon_comparison.csv"
    )

    pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key != "future_distribution"
            }
            for result in results
        ]
    ).to_csv(
        output_csv,
        index=False,
    )

    print("\nArtifacts:")
    print(f"  {output_json}")
    print(f"  {output_csv}")

    print("\nFuture-target diagnostic complete.")


if __name__ == "__main__":
    main()