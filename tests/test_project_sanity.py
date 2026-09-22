import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "03_Data"
SOURCE_DIR = ROOT / "04_Source_Code"
DASHBOARD_DIR = ROOT / "02_Streamlit_Dashboard"
MODEL_DIR = DASHBOARD_DIR / "models"
OUTPUT_DIR = DASHBOARD_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

RAW_DATA = DATA_DIR / "Thales_Group_Manufacturing.csv"
FEATURE_DATA = DATA_DIR / "manufacturing_features.parquet"

REQUIRED_RAW_COLUMNS = {
    "Machine_ID",
    "Date",
    "Timestamp",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Operation_Mode",
    "Efficiency_Status",
}


class TestProjectSanity(unittest.TestCase):

    def test_required_project_files_exist(self):
        required = [
            ROOT / "README.md",
            ROOT / "requirements.txt",
            ROOT / ".gitignore",
            SOURCE_DIR / "01_eda.py",
            SOURCE_DIR / "02_feature_engineering.py",
            SOURCE_DIR / "03_modeling.py",
            SOURCE_DIR / "04_explainability.py",
            SOURCE_DIR / "05_streamlit_app.py",
            SOURCE_DIR / "config.py",
            DASHBOARD_DIR / "app" / "app.py",
            RAW_DATA,
            FEATURE_DATA,
        ]

        missing = [
            str(path.relative_to(ROOT))
            for path in required
            if not path.exists()
        ]

        self.assertEqual(
            missing,
            [],
            f"Missing required files: {missing}",
        )

    def test_raw_dataset_schema(self):
        df = pd.read_csv(RAW_DATA)

        self.assertEqual(len(df), 100000)

        self.assertTrue(
            REQUIRED_RAW_COLUMNS.issubset(df.columns)
        )

        self.assertEqual(
            int(df.duplicated().sum()),
            0,
        )

        self.assertEqual(
            int(df.isna().sum().sum()),
            0,
        )

    def test_target_classes(self):
        df = pd.read_csv(RAW_DATA)

        self.assertEqual(
            set(df["Efficiency_Status"].unique()),
            {"Low", "Medium", "High"},
        )

    def test_feature_dataset_exists_and_is_complete(self):
        df = pd.read_parquet(FEATURE_DATA)

        required_engineered = {
            "Datetime",
            "Sensor_Stability_Score",
            "Energy_Efficiency_Ratio",
            "Error_to_Output_Ratio",
            "Quality_Adjusted_Error",
            "Network_Reliability_Score",
            "Hour",
            "Is_Weekend",
        }

        self.assertTrue(
            required_engineered.issubset(df.columns)
        )

        self.assertEqual(
            len(df),
            100000,
        )

    def test_model_artifacts_exist(self):
        required = [
            MODEL_DIR / "label_encoder.joblib",
            MODEL_DIR / "logistic_regression_model.joblib",
            MODEL_DIR / "random_forest_model.joblib",
            MODEL_DIR / "scaler.joblib",
            MODEL_DIR / "xgboost_model.json",
            MODEL_DIR / "model_meta.json",
        ]

        missing = [
            path.name
            for path in required
            if not path.exists()
        ]

        self.assertEqual(
            missing,
            [],
            f"Missing model artifacts: {missing}",
        )

    def test_model_metadata_contract(self):
        meta = json.loads(
            (MODEL_DIR / "model_meta.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            meta["task"],
            "Current-state Efficiency_Status classification benchmark",
        )

        self.assertFalse(
            meta["future_prediction_supported"]
        )

        self.assertEqual(
            meta["primary_operational_method"],
            "Transparent Business Rule",
        )

        self.assertEqual(
            set(meta["class_names"]),
            {"High", "Low", "Medium"},
        )

        self.assertGreater(
            len(meta["feature_columns"]),
            0,
        )

    def test_serialized_sklearn_models_load(self):
        joblib.load(
            MODEL_DIR
            / "logistic_regression_model.joblib"
        )

        joblib.load(
            MODEL_DIR
            / "random_forest_model.joblib"
        )

        joblib.load(
            MODEL_DIR
            / "scaler.joblib"
        )

        joblib.load(
            MODEL_DIR
            / "label_encoder.joblib"
        )

    def test_final_model_comparison_exists(self):
        path = (
            METRICS_DIR
            / "final_model_comparison.csv"
        )

        self.assertTrue(path.exists())

        results = pd.read_csv(path)

        required_methods = {
            "Majority Baseline",
            "Transparent Business Rule",
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        }

        self.assertEqual(
            set(results["method"]),
            required_methods,
        )

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]:
            self.assertIn(
                metric,
                results.columns,
            )

    def test_ablation_results_exist(self):
        path = (
            METRICS_DIR
            / "feature_ablation_results.csv"
        )

        self.assertTrue(path.exists())

        results = pd.read_csv(path)

        required_experiments = {
            "all_features",
            "no_direct_rule_features",
            "no_rule_linked_features",
            "sensor_network_only",
        }

        self.assertEqual(
            set(results["experiment"]),
            required_experiments,
        )

    def test_target_diagnostic_exists(self):
        path = (
            DIAGNOSTICS_DIR
            / "target_diagnostics_summary.json"
        )

        self.assertTrue(path.exists())

        result = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        self.assertLessEqual(
            result["business_rule_disagreements"],
            2,
        )

        self.assertGreaterEqual(
            result[
                "business_rule_full_dataset"
            ]["accuracy"],
            0.9999,
        )

    def test_temporal_diagnostic_rejects_future_claim(self):
        path = (
            DIAGNOSTICS_DIR
            / "temporal_signal_diagnostics.json"
        )

        self.assertTrue(path.exists())

        result = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        horizon_one = result["horizons"][0]

        self.assertLess(
            abs(
                horizon_one[
                    "cohen_kappa"
                ]
            ),
            0.01,
        )

        self.assertLess(
            abs(
                horizon_one[
                    "agreement_above_chance"
                ]
            ),
            0.01,
        )

    def test_dashboard_and_source_copy_match(self):
        app = (
            DASHBOARD_DIR
            / "app"
            / "app.py"
        ).read_text(
            encoding="utf-8"
        )

        source_copy = (
            SOURCE_DIR
            / "05_streamlit_app.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertEqual(
            app,
            source_copy,
            "Dashboard app and source copy have diverged.",
        )


if __name__ == "__main__":
    unittest.main()