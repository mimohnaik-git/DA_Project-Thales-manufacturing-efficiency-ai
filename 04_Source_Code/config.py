from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "03_Data"
SOURCE_DIR = PROJECT_ROOT / "04_Source_Code"
FIGURES_DIR = PROJECT_ROOT / "05_Figures"

DASHBOARD_DIR = PROJECT_ROOT / "02_Streamlit_Dashboard"
DASHBOARD_DATA_DIR = DASHBOARD_DIR / "data"
MODEL_DIR = DASHBOARD_DIR / "models"
OUTPUT_DIR = DASHBOARD_DIR / "outputs"

VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "validation"
DIAGNOSTICS_OUTPUT_DIR = OUTPUT_DIR / "diagnostics"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"

RAW_DATA_PATH = DATA_DIR / "Thales_Group_Manufacturing.csv"
FEATURE_DATA_PATH = DATA_DIR / "manufacturing_features.parquet"


def ensure_output_directories() -> None:
    """Create required project output directories."""
    for directory in (
        FIGURES_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        VALIDATION_OUTPUT_DIR,
        DIAGNOSTICS_OUTPUT_DIR,
        METRICS_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)