"""
02_feature_engineering.py

Builds engineered features used by the current-state manufacturing
efficiency analysis and benchmark models.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


SOURCE_DIR = Path(__file__).resolve().parent

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


from config import (
    DASHBOARD_DATA_DIR,
    FEATURE_DATA_PATH,
    RAW_DATA_PATH,
    ensure_output_directories,
)


ROLLING_WINDOW = 5


def rolling_cv(
    series: pd.Series,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """
    Rolling coefficient of variation.

    Higher values represent greater short-term
    variability relative to the rolling mean.
    """

    rolling_mean = series.rolling(
        window=window,
        min_periods=2,
    ).mean()

    rolling_std = series.rolling(
        window=window,
        min_periods=2,
    ).std()

    return (
        rolling_std
        / rolling_mean.replace(
            0,
            np.nan,
        )
    ).fillna(0)


def main() -> None:
    ensure_output_directories()

    DASHBOARD_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Timestamp"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="raise",
    )

    df = (
        df.sort_values(
            [
                "Machine_ID",
                "Datetime",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # Sensor stability
    # --------------------------------------------------------------
    df[
        "Temp_Stability_Index"
    ] = (
        df.groupby(
            "Machine_ID"
        )[
            "Temperature_C"
        ]
        .transform(
            rolling_cv
        )
    )

    df[
        "Vibration_Stability_Index"
    ] = (
        df.groupby(
            "Machine_ID"
        )[
            "Vibration_Hz"
        ]
        .transform(
            rolling_cv
        )
    )

    df[
        "Sensor_Stability_Score"
    ] = (
        1
        - (
            (
                df[
                    "Temp_Stability_Index"
                ]
                + df[
                    "Vibration_Stability_Index"
                ]
            )
            / 2
        ).clip(
            0,
            1,
        )
    )

    # --------------------------------------------------------------
    # Energy-efficiency ratio
    # --------------------------------------------------------------
    df[
        "Energy_Efficiency_Ratio"
    ] = (
        df[
            "Production_Speed_units_per_hr"
        ]
        / df[
            "Power_Consumption_kW"
        ].replace(
            0,
            np.nan,
        )
    ).fillna(0)

    # --------------------------------------------------------------
    # Quality / production transformations
    # --------------------------------------------------------------
    df[
        "Error_to_Output_Ratio"
    ] = (
        df[
            "Error_Rate_%"
        ]
        / df[
            "Production_Speed_units_per_hr"
        ].replace(
            0,
            np.nan,
        )
        * 1000
    ).fillna(0)

    df[
        "Quality_Adjusted_Error"
    ] = (
        df[
            "Error_Rate_%"
        ]
        + df[
            "Quality_Control_Defect_Rate_%"
        ]
    )

    # --------------------------------------------------------------
    # Network reliability score
    # --------------------------------------------------------------
    latency_min = df[
        "Network_Latency_ms"
    ].min()

    latency_max = df[
        "Network_Latency_ms"
    ].max()

    loss_min = df[
        "Packet_Loss_%"
    ].min()

    loss_max = df[
        "Packet_Loss_%"
    ].max()

    latency_range = (
        latency_max
        - latency_min
    )

    loss_range = (
        loss_max
        - loss_min
    )

    latency_normalized = (
        (
            df[
                "Network_Latency_ms"
            ]
            - latency_min
        )
        / latency_range
        if latency_range != 0
        else 0
    )

    loss_normalized = (
        (
            df[
                "Packet_Loss_%"
            ]
            - loss_min
        )
        / loss_range
        if loss_range != 0
        else 0
    )

    df[
        "Network_Reliability_Score"
    ] = (
        1
        - (
            0.5
            * latency_normalized
            + 0.5
            * loss_normalized
        )
    ) * 100

    # --------------------------------------------------------------
    # Time features
    # --------------------------------------------------------------
    df[
        "Hour"
    ] = df[
        "Datetime"
    ].dt.hour

    df[
        "DayOfWeek"
    ] = df[
        "Datetime"
    ].dt.dayofweek

    df[
        "Is_Weekend"
    ] = (
        df[
            "DayOfWeek"
        ]
        .isin(
            [
                5,
                6,
            ]
        )
        .astype(int)
    )

    # --------------------------------------------------------------
    # Save authoritative engineered dataset
    # --------------------------------------------------------------
    df.to_parquet(
        FEATURE_DATA_PATH,
        index=False,
    )

    # Dashboard gets an identical deployment copy.
    dashboard_feature_path = (
        DASHBOARD_DATA_DIR
        / "manufacturing_features.parquet"
    )

    df.to_parquet(
        dashboard_feature_path,
        index=False,
    )

    print(
        "=" * 72
    )

    print(
        "FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print(
        f"\nSaved:\n"
        f"  {FEATURE_DATA_PATH}\n"
        f"  {dashboard_feature_path}"
    )

    print(
        "\nEngineered-feature summary:"
    )

    print(
        df[
            [
                "Sensor_Stability_Score",
                "Energy_Efficiency_Ratio",
                "Error_to_Output_Ratio",
                "Network_Reliability_Score",
            ]
        ]
        .describe()
        .round(4)
    )


if __name__ == "__main__":
    main()