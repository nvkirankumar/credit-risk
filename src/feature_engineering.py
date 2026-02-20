"""Feature engineering helpers for credit risk modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_MAP = {
    "unknown": 0,
    "little": 1,
    "moderate": 2,
    "quite rich": 3,
    "rich": 4,
}


HIGH_RISK_PURPOSES = {
    "education",
    "business",
    "repairs",
    "vacation/others",
}


def add_credit_duration_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized loan burden proxy based on amount and duration."""
    data = df.copy()
    duration = data["Duration"].replace(0, np.nan)
    data["Credit_to_Duration_Ratio"] = data["Credit amount"] / duration
    data["Credit_to_Duration_Ratio"] = data["Credit_to_Duration_Ratio"].fillna(0.0)
    return data


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket applicant age into risk-relevant life-stage segments."""
    data = df.copy()
    bins = [17, 25, 35, 50, 120]
    labels = ["18-25", "26-35", "36-50", "50+"]
    data["Age_Group"] = pd.cut(data["Age"], bins=bins, labels=labels)
    data["Age_Group"] = data["Age_Group"].astype(str)
    return data


def add_account_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Combine savings/checking levels into a single stability score."""
    data = df.copy()
    savings_score = data["Saving accounts"].map(SCORE_MAP).fillna(0)
    checking_score = data["Checking account"].map(SCORE_MAP).fillna(0)
    data["Account_Stability"] = savings_score + checking_score
    return data


def add_high_risk_purpose_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag purposes typically associated with higher default uncertainty."""
    data = df.copy()
    normalized_purpose = data["Purpose"].astype(str).str.lower()
    data["High_Risk_Purpose"] = normalized_purpose.isin(HIGH_RISK_PURPOSES).astype(int)
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in a deterministic order."""
    data = df.copy()
    data = add_credit_duration_ratio(data)
    data = add_age_group(data)
    data = add_account_stability(data)
    data = add_high_risk_purpose_flag(data)
    return data