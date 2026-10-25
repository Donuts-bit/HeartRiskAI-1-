"""Clinical Model Evaluation Module for HeartRiskAI.
Computes comprehensive performance metrics tailored to imbalanced clinical data:
Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score, ROC-AUC, PR-AUC,
Brier score, and Confusion Matrix.
Renders academic-quality ROC curves, PR curves, and confusion matrix heatmaps.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.config import REPORTS_DIR, FIGURES_DIR


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
    experiment_name: str = "Default",
    feature_count: int = 15,
    training_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for clinical binary classification.
    
    Includes Specificity (True Negative Rate) alongside Sensitivity (Recall) and PR-AUC,
    which are essential for clinical risk triage under class imbalance.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    y_pr = np.asarray(y_prob).ravel()

    # Confusion matrix extraction
    cm = confusion_matrix(y_t, y_p)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: single class predicted
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_t)) == 1:
            if y_t[0] == 0:
                tn = len(y_t)
            else:
                tp = len(y_t)

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_t, y_pr))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, y_pr))
    except Exception:
        pr_auc = float(np.mean(y_t))

    try:
        brier = float(brier_score_loss(y_t, y_pr))
    except Exception:
        brier = 1.0

    return {
        "experiment": experiment_name,
        "model": model_name,
        "features_count": feature_count,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
        "training_time_sec": round(training_time_sec, 3),
    }


def plot_roc_curves(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plots multi-model ROC curves with AUC annotations."""
    if output_path is None:
        output_path = FIGURES_DIR / "roc_curves_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#e6ab02", "#a6761d", "#666666"]

    for idx, (label, data) in enumerate(results_dict.items()):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_pr_curves(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plots multi-model Precision-Recall curves with Average Precision annotations."""
    if output_path is None:
        output_path = FIGURES_DIR / "pr_curves_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#e6ab02", "#a6761d", "#666666"]

    baseline_set = False
    for idx, (label, data) in enumerate(results_dict.items()):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)
        color = colors[idx % len(colors)]
        ax.plot(rec, prec, color=color, lw=2, label=f"{label} (PR-AUC = {pr_auc_val:.3f})")

        if not baseline_set:
            prevalence = float(np.mean(y_true))
            ax.axhline(y=prevalence, color="gray", linestyle="--", lw=1.2, label=f"Prevalence Baseline ({prevalence:.1%})")
            baseline_set = True

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("Precision-Recall (PR) Curves (Imbalanced Clinical Cohort)", fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    results_list: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plots grid of confusion matrices annotated with raw counts and normalized rates."""
    if output_path is None:
        output_path = FIGURES_DIR / "confusion_matrices.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(results_list)
    n_cols = 4 if n_models >= 4 else n_models
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.4 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No CHD", "CHD"],
            yticklabels=["No CHD", "CHD"],
        )
        title_str = f"{res['model']}\n({res['experiment']})"
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

    # Hide extra axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
