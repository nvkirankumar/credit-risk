"""Evaluation and visualization utilities for classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "AUC_ROC": float(roc_auc_score(y_test, y_score)),
    }


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create sortable comparison table from model metrics."""
    df = pd.DataFrame(results).T
    return df.sort_values(by=["Recall", "F1"], ascending=False)


def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """Plot and optionally save confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(
    model_dict: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | Path | None = None,
) -> None:
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(8, 6))
    for name, model in model_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "Precision-Recall Curve",
    save_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot precision-recall curve and return arrays."""
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.predict(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()

    return precision, recall, thresholds


def find_threshold_for_target_recall(
    y_true: pd.Series,
    y_scores: np.ndarray,
    min_recall: float = 0.75,
) -> Tuple[float, float, float]:
    """Find a probability threshold that satisfies minimum recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    valid = np.where(recall[:-1] >= min_recall)[0]
    if len(valid) == 0:
        return 0.5, float(precision[-1]), float(recall[-1])

    # Select threshold with best precision among those meeting recall target.
    idx = valid[np.argmax(precision[valid])]
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx])