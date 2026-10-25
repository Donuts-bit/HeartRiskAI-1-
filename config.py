"""Central Configuration for HeartRiskAI.
Defines directory paths, feature schemas, clinical thresholds, and random seeds.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "framingham.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Schema Constants
TARGET_COLUMN = "TenYearCHD"

CATEGORICAL_FEATURES: List[str] = [
    "male",
    "education",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]

NUMERICAL_FEATURES: List[str] = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

ALL_FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
EXPECTED_COLUMNS: List[str] = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]

# Clinical Plausibility Ranges for Data Quality Auditing
PHYSIOLOGICAL_BOUNDS: Dict[str, Dict[str, float]] = {
    "age": {"min": 18.0, "max": 100.0},
    "cigsPerDay": {"min": 0.0, "max": 100.0},
    "totChol": {"min": 80.0, "max": 700.0},
    "sysBP": {"min": 60.0, "max": 300.0},
    "diaBP": {"min": 40.0, "max": 200.0},
    "BMI": {"min": 10.0, "max": 70.0},
    "heartRate": {"min": 30.0, "max": 220.0},
    "glucose": {"min": 30.0, "max": 500.0},
}

# Clinical Thresholds (ACC/AHA and NCEP ATP III Guidelines)
CLINICAL_THRESHOLDS = {
    "sysBP": {
        "elevated": 120.0,
        "stage1_htn": 130.0,
        "stage2_htn": 140.0,
        "crisis_htn": 180.0,
    },
    "diaBP": {
        "stage1_htn": 80.0,
        "stage2_htn": 90.0,
        "crisis_htn": 120.0,
    },
    "totChol": {
        "borderline": 200.0,
        "high": 240.0,
    },
    "glucose": {
        "prediabetes": 100.0,
        "diabetes": 126.0,
    },
    "BMI": {
        "overweight": 25.0,
        "obese": 30.0,
    },
}

# Mandatory Clinical Features (must be retained across all experiments)
MANDATORY_FEATURES: List[str] = ["age", "male"]

# Model and Preprocessor Persistence Paths
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"

# Experimentation & Reproducibility
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20
CV_FOLDS = 5
