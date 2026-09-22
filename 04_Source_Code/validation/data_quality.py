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
    RAW_DATA_PATH,
    ensure_output_directories,
)


TARGET = "Efficiency_Status"


def main() -> None:
    ensure_output_directories()

    df = pd.read_csv(RAW_DATA_PATH)

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": {
            column: int(value)
            for column, value in df.isna().sum().items()
        },
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    if TARGET in df.columns:
        counts = df[TARGET].value_counts(dropna=False)

        report["target_distribution"] = {
            str(label): {
                "count": int(count),
                "percentage": round(float(count / len(df) * 100), 4),
            }
            for label, count in counts.items()
        }

    output_path = DIAGNOSTICS_OUTPUT_DIR / "data_quality_report.json"

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("DATA QUALITY AUDIT")
    print("=" * 70)
    print(f"Rows:           {report['rows']:,}")
    print(f"Columns:        {report['columns']}")
    print(f"Duplicates:     {report['duplicate_rows']:,}")
    print(f"Missing values: {report['missing_values_total']:,}")

    if "target_distribution" in report:
        print("\nTarget distribution:")

        for label, values in report["target_distribution"].items():
            print(
                f"{label:<10}"
                f"{values['count']:>10,}"
                f"  ({values['percentage']:.2f}%)"
            )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()