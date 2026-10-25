"""Multi-Model Training, Hyperparameter Tuning, and Experimentation for HeartRiskAI.
Trains and compares Logistic Regression, Random Forest, SVM, and XGBoost under two regimes:
- Experiment A: All permitted input features (15 clinical features).
- Experiment B: Genetic Algorithm selected features (6 parsimonious features).

Enforces strict leakage-free protocols:
1. Identical stratified training/test split partitions.
2. Stratified Cross-Validation on training data only.
3. Class imbalance handled via class weights and XGBoost scale_pos_weight.
4. Sealed test set evaluated ONLY after model selection is completed.
5. All results saved to reports/model_comparison.csv (no fabricated metrics).
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from xgboost import XGBClassifier

from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    SELECTED_FEATURES_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.evaluate_models import (
    calculate_classification_metrics,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
)

logger = logging.getLogger(__name__)


def get_model_specifications(scale_pos_weight: float) -> Dict[str, Dict[str, Any]]:
    """Defines candidate classifiers with balanced loss weighting and parameter grids."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "C": [0.05, 0.2, 1.0, 5.0],
                "penalty": ["l1", "l2"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "min_samples_split": [5, 10],
            },
        },
        "SVM": {
            "estimator": SVC(
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "C": [0.1, 1.0, 3.0],
                "kernel": ["rbf"],
                "gamma": ["scale"],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            "param_grid": {
                "n_estimators": [100, 150],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
            },
        },
    }


def tune_and_train_candidate(
    model_name: str,
    spec: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 3,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tunes hyperparameters using training-data-only GridSearchCV."""
    start_time = time.time()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=spec["estimator"],
        param_grid=spec["param_grid"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    return grid_search.best_estimator_, grid_search.best_params_, elapsed


def evaluate_cross_validation(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    experiment_name: str,
    training_time: float,
    cv_folds: int = CV_FOLDS,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Generates out-of-fold cross-validation probability predictions and metrics."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    y_pred_oof = (y_prob_oof >= 0.5).astype(int)

    metrics = calculate_classification_metrics(
        y_true=y_train,
        y_pred=y_pred_oof,
        y_prob=y_prob_oof,
        model_name=model_name,
        experiment_name=experiment_name,
        feature_count=X_train.shape[1],
        training_time_sec=training_time,
    )
    return metrics, y_prob_oof


def run_experiments(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    selected_features_path: Path = SELECTED_FEATURES_PATH,
) -> Dict[str, Any]:
    """Executes Experiment A (All Features) and Experiment B (GA Features)."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_all = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test_all = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # Calculate scale_pos_weight for XGBoost from y_train
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    # Load GA selected features
    with open(selected_features_path, "r", encoding="utf-8") as f:
        ga_metadata = json.load(f)
    ga_features = ga_metadata["selected_features"]

    X_train_ga = X_train_all[ga_features]
    X_test_ga = X_test_all[ga_features]

    print(f"Dataset partitions loaded: Train={X_train_all.shape[0]}, Test={X_test_all.shape[0]}")
    print(f"Experiment A features (All): {X_train_all.shape[1]} features")
    print(f"Experiment B features (GA):  {X_train_ga.shape[1]} features: {ga_features}")
    print(f"XGBoost scale_pos_weight:    {scale_pos_weight:.2f}")

    experiments = [
        ("Experiment A - All Features", X_train_all, X_test_all),
        ("Experiment B - GA Selected", X_train_ga, X_test_ga),
    ]

    all_cv_results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    oof_predictions_dict: Dict[str, Dict[str, Any]] = {}

    for exp_name, X_tr, X_te in experiments:
        print("\n" + "=" * 70)
        print(f"Running {exp_name} ({X_tr.shape[1]} features)")
        print("=" * 70)

        model_specs = get_model_specifications(scale_pos_weight)

        for model_name, spec in model_specs.items():
            print(f" -> Tuning & Training: {model_name}...")
            best_model, best_params, train_time = tune_and_train_candidate(
                model_name, spec, X_tr, y_train, cv_folds=3
            )
            key = f"{model_name} ({exp_name.split(' - ')[1]})"
            trained_models[key] = {
                "model": best_model,
                "best_params": best_params,
                "experiment": exp_name,
                "features": list(X_tr.columns),
            }

            # Out-of-fold Stratified CV evaluation on training set
            cv_metrics, y_prob_oof = evaluate_cross_validation(
                best_model, X_tr, y_train, model_name, exp_name, train_time, cv_folds=5
            )
            all_cv_results.append(cv_metrics)
            oof_predictions_dict[key] = {
                "y_true": y_train.values,
                "y_prob": y_prob_oof,
            }

            print(
                f"    [CV Results] ROC-AUC: {cv_metrics['roc_auc']:.4f} | "
                f"PR-AUC: {cv_metrics['pr_auc']:.4f} | "
                f"Sensitivity: {cv_metrics['recall_sensitivity']:.4f} | "
                f"Specificity: {cv_metrics['specificity']:.4f} | "
                f"Time: {train_time:.2f}s"
            )

    # 1. Compile and save Cross-Validation Model Comparison
    cv_results_df = pd.DataFrame(all_cv_results)
    comparison_csv_path = REPORTS_DIR / "model_comparison.csv"
    cv_results_df.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved cross-validation benchmark results to {comparison_csv_path}")

    # 2. Select Winning Model based on CV ROC-AUC and PR-AUC
    # In imbalanced clinical risk prediction, ROC-AUC and PR-AUC dictate triage efficacy
    cv_results_df["selection_score"] = cv_results_df["roc_auc"] + cv_results_df["pr_auc"]
    best_row = cv_results_df.loc[cv_results_df["selection_score"].idxmax()]
    winning_model_name = best_row["model"]
    winning_exp_name = best_row["experiment"]
    winning_key = f"{winning_model_name} ({winning_exp_name.split(' - ')[1]})"

    print("\n" + "*" * 70)
    print(f"WINNING MODEL SELECTED (TRAINING CV): {winning_key}")
    print(f"CV ROC-AUC: {best_row['roc_auc']:.4f} | CV PR-AUC: {best_row['pr_auc']:.4f}")
    print(f"CV Sensitivity: {best_row['recall_sensitivity']:.4f} | CV Specificity: {best_row['specificity']:.4f}")
    print("*" * 70)

    # 3. Final Evaluation on Held-Out Test Set (Sealed until this point)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST SET (N = 848)")
    print("=" * 70)

    test_eval_results: List[Dict[str, Any]] = []
    test_predictions_dict: Dict[str, Dict[str, Any]] = {}

    for key, item in trained_models.items():
        m = item["model"]
        feats = item["features"]
        exp = item["experiment"]
        m_name = key.split(" (")[0]

        X_te = X_test_all[feats]
        y_prob_test = m.predict_proba(X_te)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)

        t_metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=y_pred_test,
            y_prob=y_prob_test,
            model_name=m_name,
            experiment_name=exp,
            feature_count=len(feats),
            training_time_sec=0.0,
        )
        test_eval_results.append(t_metrics)
        test_predictions_dict[key] = {
            "y_true": y_test.values,
            "y_prob": y_prob_test,
        }

        print(
            f"{key:<32} | Test ROC-AUC: {t_metrics['roc_auc']:.4f} | "
            f"Test PR-AUC: {t_metrics['pr_auc']:.4f} | "
            f"Sensitivity: {t_metrics['recall_sensitivity']:.4f} | "
            f"Specificity: {t_metrics['specificity']:.4f}"
        )

    test_results_df = pd.DataFrame(test_eval_results)
    test_csv_path = REPORTS_DIR / "test_evaluation_summary.csv"
    test_results_df.to_csv(test_csv_path, index=False)

    # 4. Serialize Winning Model & Candidate Models
    winning_model_obj = trained_models[winning_key]["model"]
    winning_model_features = trained_models[winning_key]["features"]
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(
        {
            "model": winning_model_obj,
            "model_name": winning_model_name,
            "experiment": winning_exp_name,
            "features": winning_model_features,
            "best_params": trained_models[winning_key]["best_params"],
            "cv_metrics": best_row.to_dict(),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
        },
        best_model_path,
    )
    print(f"\nPersisted winning model pipeline to {best_model_path}")

    # Also save each individual candidate model for explainability / analysis
    for key, item in trained_models.items():
        slug = key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        cand_path = MODELS_DIR / f"{slug}.joblib"
        joblib.dump(item, cand_path)

    # 5. Generate Academic Plots
    print("\nRendering comparative ROC, PR, and Confusion Matrix figures...")
    plot_roc_curves(test_predictions_dict, FIGURES_DIR / "roc_curves_comparison.png")
    plot_pr_curves(test_predictions_dict, FIGURES_DIR / "pr_curves_comparison.png")
    plot_confusion_matrices(test_eval_results, FIGURES_DIR / "confusion_matrices.png")
    print("Figures saved to reports/figures/")

    return {
        "cv_results_df": cv_results_df,
        "test_results_df": test_results_df,
        "winning_key": winning_key,
        "winning_model_path": best_model_path,
    }


def main():
    """CLI execution entrypoint."""
    print("=" * 70)
    print("HeartRiskAI: Classifier Training & Experimental Evaluation")
    print("=" * 70)
    results = run_experiments()
    print("\nTraining and evaluation pipeline executed successfully.")


if __name__ == "__main__":
    main()
