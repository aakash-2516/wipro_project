"""
Shared helpers used by both train_model.py and app.py
so training and prediction always use the exact same preprocessing.
"""
import pandas as pd

DATA_PATH = "data/garments_worker_productivity.csv"
TARGET = "actual_productivity"

CATEGORICAL = ["department", "quarter", "day", "team"]

NUMERIC = [
    "targeted_productivity", "smv", "wip", "over_time", "incentive",
    "idle_time", "idle_men", "no_of_style_change", "no_of_workers",
    # engineered
    "overtime_per_worker", "has_incentive",
]

FEATURES = CATEGORICAL + NUMERIC

# Assumption used to turn productivity into an estimated unit count.
# A standard shift = 8 hours = 480 minutes per worker.
STANDARD_SHIFT_MIN = 480


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quality issues in the raw UCI file."""
    df = df.copy()
    # 'sweing' typo and 'finishing ' (trailing space) duplicates
    df["department"] = (df["department"].str.strip()
                        .replace({"sweing": "sewing"}))
    # WIP is missing for every finishing-department row (no WIP is tracked there)
    df["wip"] = df["wip"].fillna(0)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["team"] = df["team"].astype(str)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features (works for a single-row prediction too)."""
    df = df.copy()
    df["overtime_per_worker"] = df["over_time"] / df["no_of_workers"].clip(lower=1)
    df["has_incentive"] = (df["incentive"] > 0).astype(int)
    return df


def load_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    return add_features(clean_data(pd.read_csv(path)))


def estimate_units(productivity: float, workers: float, over_time_min: float,
                   smv: float) -> int:
    """
    Industrial-engineering efficiency formula, rearranged:

        efficiency = (units x SMV) / available_minutes
        units      = efficiency x available_minutes / SMV

    available_minutes = workers x 480 (standard shift) + overtime minutes.
    This is an ESTIMATE - the dataset does not record unit counts.
    """
    available = workers * STANDARD_SHIFT_MIN + over_time_min
    return int(round(productivity * available / max(smv, 0.1)))
