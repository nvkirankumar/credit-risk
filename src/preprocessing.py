"""Data loading, validation, preprocessing, and split utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_MAP = {"good": 0, "bad": 1}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load credit risk dataset from CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return pd.read_csv(csv_path)


def basic_data_report(df: pd.DataFrame) -> Dict[str, object]:
    """Return core dataset diagnostics used in EDA."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_target(
    df: pd.DataFrame,
    target_col: str = "Risk",
    mapping: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """Encode target labels to numeric values."""
    data = df.copy()
    label_map = mapping or TARGET_MAP

    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not present in dataframe.")

    data[target_col] = data[target_col].astype(str).str.lower().map(label_map)
    if data[target_col].isna().any():
        raise ValueError("Target encoding produced NaN values. Check label mapping.")

    data[target_col] = data[target_col].astype(int)
    return data


def fill_missing_accounts(
    df: pd.DataFrame,
    account_columns: Tuple[str, str] = ("Saving accounts", "Checking account"),
    fill_value: str = "unknown",
) -> pd.DataFrame:
    """Impute missing account status fields with a categorical bucket."""
    data = df.copy()
    for col in account_columns:
        if col in data.columns:
            data[col] = data[col].fillna(fill_value)
    return data


def split_data(
    df: pd.DataFrame,
    target_col: str = "Risk",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create stratified train/test split for classification."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: str | Path = "data/processed",
) -> None:
    """Persist train/test datasets to disk."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output / "X_train.csv", index=False)
    X_test.to_csv(output / "X_test.csv", index=False)
    y_train.to_frame(name="Risk").to_csv(output / "y_train.csv", index=False)
    y_test.to_frame(name="Risk").to_csv(output / "y_test.csv", index=False)