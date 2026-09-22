"""
01_eda.py

Exploratory Data Analysis for the Thales Manufacturing Efficiency project.

Outputs:
- Figures 01-07
- EDA summary JSON
- Rule-discovery contingency table
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


SOURCE_DIR = Path(__file__).resolve().parent

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


from config import (
    FIGURES_DIR,
    OUTPUT_DIR,
    RAW_DATA_PATH,
    ensure_output_directories,
)


NUMERIC_COLUMNS = [
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

STATUS_ORDER = [
    "Low",
    "Medium",
    "High",
]

STATUS_COLORS = {
    "Low": "#d62728",
    "Medium": "#ff9f1c",
    "High": "#2ca02c",
}


def load_data() -> pd.DataFrame:
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

    return (
        df.sort_values("Datetime")
        .reset_index(drop=True)
    )


def build_summary(
    df: pd.DataFrame,
) -> dict:

    machine_efficiency = (
        df.groupby("Machine_ID")[
            "Efficiency_Status"
        ]
        .apply(
            lambda values: (
                values == "Low"
            ).mean()
            * 100
        )
        .sort_values(
            ascending=False
        )
    )

    return {
        "n_rows": int(
            len(df)
        ),
        "n_machines": int(
            df["Machine_ID"].nunique()
        ),
        "date_range": [
            str(
                df["Datetime"].min()
            ),
            str(
                df["Datetime"].max()
            ),
        ],
        "class_counts": {
            str(key): int(value)
            for key, value
            in (
                df["Efficiency_Status"]
                .value_counts()
                .items()
            )
        },
        "class_pct": {
            str(key): float(value)
            for key, value
            in (
                df["Efficiency_Status"]
                .value_counts(
                    normalize=True
                )
                .mul(100)
                .round(2)
                .items()
            )
        },
        "operation_mode_counts": {
            str(key): int(value)
            for key, value
            in (
                df["Operation_Mode"]
                .value_counts()
                .items()
            )
        },
        "missing_values": int(
            df.isna()
            .sum()
            .sum()
        ),
        "duplicate_rows": int(
            df.duplicated()
            .sum()
        ),
        "machine_low_pct_range": [
            float(
                machine_efficiency.min()
            ),
            float(
                machine_efficiency.max()
            ),
        ],
    }


def main() -> None:
    ensure_output_directories()

    sns.set_theme(
        style="whitegrid",
        palette="deep",
    )

    plt.rcParams[
        "figure.dpi"
    ] = 150

    df = load_data()

    summary = build_summary(
        df
    )

    # --------------------------------------------------------------
    # Figure 01 — Class distribution
    # --------------------------------------------------------------
    figure, axis = plt.subplots(
        figsize=(6, 4.5)
    )

    counts = (
        df["Efficiency_Status"]
        .value_counts()
        .reindex(
            STATUS_ORDER
        )
    )

    bars = axis.bar(
        counts.index,
        counts.to_numpy(dtype=float),
        color=[
            STATUS_COLORS[
                status
            ]
            for status
            in counts.index
        ],
    )

    for bar in bars:
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{int(bar.get_height()):,}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    axis.set_title(
        "Class Distribution: Efficiency_Status",
        fontsize=13,
        fontweight="bold",
    )

    axis.set_ylabel(
        "Number of Records"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "01_class_distribution.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Figure 02 — Correlation matrix
    # --------------------------------------------------------------
    figure, axis = plt.subplots(
        figsize=(8, 6.5)
    )

    correlation = df[
        NUMERIC_COLUMNS
    ].corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=axis,
        cbar_kws={
            "label": (
                "Pearson correlation"
            )
        },
    )

    axis.set_title(
        "Correlation Matrix — Sensor, Production & Network Features",
        fontsize=12,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "02_correlation_heatmap.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Figure 03 — Rule-linked distributions
    # --------------------------------------------------------------
    key_features = [
        "Error_Rate_%",
        "Production_Speed_units_per_hr",
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
    )

    for axis, feature in zip(
        axes,
        key_features,
    ):

        for status in STATUS_ORDER:

            sns.kdeplot(
                x=df.loc[
                    df[
                        "Efficiency_Status"
                    ]
                    == status,
                    feature,
                ],
                ax=axis,
                label=status,
                color=STATUS_COLORS[
                    status
                ],
                fill=True,
                alpha=0.25,
                linewidth=2,
            )

        axis.set_title(
            f"Distribution of {feature} by Efficiency Status",
            fontsize=11,
            fontweight="bold",
        )

        axis.legend(
            title="Efficiency"
        )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "03_key_driver_distributions.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Figure 04 — Sensor/network separation
    # --------------------------------------------------------------
    descriptive_features = [
        "Temperature_C",
        "Vibration_Hz",
        "Network_Latency_ms",
        "Packet_Loss_%",
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    for axis, feature in zip(
        axes.flat,
        descriptive_features,
    ):

        sns.boxplot(
            data=df,
            x="Efficiency_Status",
            y=feature,
            order=STATUS_ORDER,
            ax=axis,
            palette=STATUS_COLORS,
        )

        axis.set_title(
            feature,
            fontsize=10,
            fontweight="bold",
        )

    figure.suptitle(
        "Sensor/Network Features Show Little Separation Across Efficiency Classes",
        fontsize=12,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "04_noise_feature_boxplots.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Figure 05 — Operation mode
    # --------------------------------------------------------------
    figure, axis = plt.subplots(
        figsize=(7, 4.8)
    )

    operation_mix = (
        pd.crosstab(
            df[
                "Operation_Mode"
            ],
            df[
                "Efficiency_Status"
            ],
            normalize="index",
        )
        .reindex(
            columns=STATUS_ORDER,
            fill_value=0,
        )
        * 100
    )

    operation_mix.plot(
        kind="bar",
        stacked=True,
        ax=axis,
        color=[
            STATUS_COLORS[
                status
            ]
            for status
            in STATUS_ORDER
        ],
    )

    axis.set_ylabel(
        "% of Records"
    )

    axis.set_title(
        "Efficiency Status Mix by Operation Mode",
        fontsize=12,
        fontweight="bold",
    )

    axis.legend(
        title="Efficiency",
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "05_efficiency_by_operation_mode.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Figure 06 — Machine-level profile
    # --------------------------------------------------------------
    machine_efficiency = (
        df.groupby("Machine_ID")[
            "Efficiency_Status"
        ]
        .apply(
            lambda values: (
                values == "Low"
            ).mean()
            * 100
        )
        .sort_values(
            ascending=False
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    machine_efficiency.plot(
        kind="bar",
        ax=axis,
        width=0.8,
    )

    axis.set_ylabel(
        "% of records classified Low"
    )

    axis.set_xlabel(
        "Machine ID"
    )

    axis.set_title(
        "Share of 'Low' Efficiency Records by Machine",
        fontsize=12,
        fontweight="bold",
    )

    axis.set_xticklabels(
        axis.get_xticklabels(),
        fontsize=6,
        rotation=90,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "06_machine_low_share.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Figure 07 — Daily efficiency mix
    # --------------------------------------------------------------
    daily = (
        df.groupby(
            [
                df[
                    "Datetime"
                ].dt.date,
                "Efficiency_Status",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=STATUS_ORDER,
            fill_value=0,
        )
    )

    daily_percentage = (
        daily.div(
            daily.sum(
                axis=1
            ),
            axis=0,
        )
        * 100
    )

    figure, axis = plt.subplots(
        figsize=(11, 4.8)
    )

    for status in STATUS_ORDER:

        axis.plot(
            daily_percentage.index,
            daily_percentage[
                status
            ],
            marker="o",
            markersize=3,
            label=status,
            color=STATUS_COLORS[
                status
            ],
        )

    axis.set_ylabel(
        "% of Records"
    )

    axis.set_title(
        "Daily Efficiency Mix Over Time",
        fontsize=12,
        fontweight="bold",
    )

    axis.legend(
        title="Efficiency"
    )

    figure.autofmt_xdate()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "07_daily_efficiency_trend.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------------
    # Descriptive target-rule contingency table
    # --------------------------------------------------------------
    error_band = pd.cut(
        df["Error_Rate_%"],
        bins=[
            -np.inf,
            2,
            5,
            np.inf,
        ],
        labels=[
            "<=2%",
            "2-5%",
            ">5%",
        ],
    )

    speed_band = pd.cut(
        df[
            "Production_Speed_units_per_hr"
        ],
        bins=[
            -np.inf,
            200,
            400,
            np.inf,
        ],
        labels=[
            "<200",
            "200-400",
            ">=400",
        ],
        right=False,
    )

    rule_table = pd.crosstab(
        [
            error_band,
            speed_band,
        ],
        df[
            "Efficiency_Status"
        ],
    )

    rule_table.to_csv(
        OUTPUT_DIR
        / "rule_discovery_table.csv"
    )

    (
        OUTPUT_DIR
        / "eda_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 72
    )

    print(
        "EDA COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()