from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "Datasets"

RAW_DATA = DATA_DIR / "raw" / "social_media_vs_productivity.csv"

PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DATA = PROCESSED_DIR / "train.csv"
TEST_DATA = PROCESSED_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)