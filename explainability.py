"""Model Explainability and Clinical Attribution Module for HeartRiskAI.
Leverages SHAP (SHapley Additive exPlanations) to provide local and global
model interpretability for cardiovascular risk predictions.

Key Principles:
1. Explains predictions from trained classifiers (Linear, Tree, and Kernel explainers).
2. Uses exact feature schemas expected by the trained pipeline.
3. Gracefully falls back if a model type is unsupported by Tree/Linear SHAP.
4. Distinguishes statistical feature attribution from rule-based expert systems.
5. Strict scientific disclaimer: SHAP values represent model-specific predictive
   associations and MUST NOT be interpreted as medical causation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TRAIN_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


class ClinicalModelExplainer:
    """Provides SHAP-based local and global model explanations."""

    CAUSALITY_DISCLAIMER = (
        "SCIENTIFIC & MEDICAL DISCLAIMER: SHAP values quantify the additive contribution "
        "of each feature to the model's statistical prediction relative to the background baseline. "
        "They reflect internal model decision boundaries and empirical correlations in the training data. "
        "SHAP values DO NOT establish biological causality, physiological etiology, or medical causation. "
        "High feature importance indicates strong predictive association within this algorithm, "
        "not a clinical mechanism of disease."
    )

    def __init__(
        self,
        model_artifact_path: Optional[Union[str, Path]] = None,
        background_data: Optional[pd.DataFrame] = None,
    ):
        path = Path(model_artifact_path) if model_artifact_path else (MODELS_DIR / "best_model.joblib")
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}. Run src.train_models first.")

        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.model_name = artifact.get("model_name", type(self.model).__name__)
        self.experiment = artifact.get("experiment", "Default")
        self.features = list(artifact.get("features", []))

        if not self.features:
            raise ValueError("Model artifact did not contain a 'features' list.")

        # Load background data for baseline expectations
        if background_data is None:
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            self.background_data = train_df[self.features].copy()
        else:
            self.background_data = background_data[self.features].copy()

        # Sample background data for computational efficiency in explainers
        n_samples = min(100, len(self.background_data))
        self.background_sample = self.background_data.sample(n=n_samples, random_state=RANDOM_STATE)

        self.explainer, self.explainer_type = self._initialize_explainer()

    def _initialize_explainer(self) -> Tuple[Any, str]:
        """Initializes the most appropriate SHAP explainer with robust fallbacks."""
        # 1. Tree Explainer for Tree Ensembles (Random Forest, XGBoost)
        model_type_name = type(self.model).__name__.lower()
        if "forest" in model_type_name or "xgb" in model_type_name or "gradient" in model_type_name:
            try:
                explainer = shap.TreeExplainer(self.model)
                return explainer, "TreeExplainer"
            except Exception as e:
                logger.warning("TreeExplainer initialization failed: %s. Falling back to KernelExplainer.", e)

        # 2. Linear Explainer for Linear Models (Logistic Regression)
        if "logistic" in model_type_name or "linear" in model_type_name:
            try:
                explainer = shap.LinearExplainer(self.model, self.background_sample)
                return explainer, "LinearExplainer"
            except Exception as e:
                logger.warning("LinearExplainer initialization failed: %s. Falling back to KernelExplainer.", e)

        # 3. Kernel Explainer (Model-agnostic fallback)
        try:
            summary_sample = shap.kmeans(self.background_sample, min(10, len(self.background_sample)))
            predict_fn = getattr(self.model, "predict_proba", self.model.predict)
            explainer = shap.KernelExplainer(predict_fn, summary_sample)
            return explainer, "KernelExplainer"
        except Exception as e:
            logger.warning("KernelExplainer initialization failed: %s. Using coefficient fallback.", e)
            return None, "CoefficientFallback"

    def format_input_instance(self, instance: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> pd.DataFrame:
        """Aligns input instance with the exact feature set and ordering expected by the model."""
        if isinstance(instance, dict):
            df = pd.DataFrame([instance])
        elif isinstance(instance, pd.Series):
            df = instance.to_frame().T
        elif isinstance(instance, pd.DataFrame):
            df = instance.copy()
        else:
            raise TypeError("Instance must be a dict, pd.Series, or pd.DataFrame.")

        missing_feats = set(self.features) - set(df.columns)
        if missing_feats:
            raise ValueError(f"Instance is missing required model features: {missing_feats}")

        # Return strictly the features in the exact expected order
        return df[self.features].copy()

    def explain_instance(
        self,
        instance: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculates local SHAP feature attributions for an individual patient instance."""
        X_inst = self.format_input_instance(instance)

        # Predict risk probability
        if hasattr(self.model, "predict_proba"):
            pred_prob = float(self.model.predict_proba(X_inst)[0, 1])
        else:
            pred_prob = float(self.model.predict(X_inst)[0])

        if self.explainer_type in ("TreeExplainer", "LinearExplainer"):
            shap_obj = self.explainer(X_inst)
            values = shap_obj.values[0]
            # Handle multi-output (e.g. TreeExplainer returning (1, features, 2))
            if values.ndim == 2 and values.shape[1] == 2:
                values = values[:, 1]
            base_val = float(shap_obj.base_values[0] if np.isscalar(shap_obj.base_values[0]) else shap_obj.base_values[0][1])
        elif self.explainer_type == "KernelExplainer":
            kernel_shap = self.explainer.shap_values(X_inst)
            if isinstance(kernel_shap, list) and len(kernel_shap) == 2:
                values = kernel_shap[1][0]
            else:
                values = kernel_shap[0]
            base_val = float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value)
        else:
            # Fallback to normalized linear coefficients * value
            coefs = getattr(self.model, "coef_", np.ones((1, len(self.features))))[0]
            values = (X_inst.values[0] * coefs).astype(float)
            base_val = float(getattr(self.model, "intercept_", [0.0])[0])

        shap_dict = {feat: round(float(val), 5) for feat, val in zip(self.features, values)}
        raw_vals_dict = {feat: round(float(X_inst.iloc[0][feat]), 3) for feat in self.features}

        # Sort features by absolute contribution
        sorted_by_abs = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_increasing = [f"{k} (+{v:.4f})" for k, v in sorted_by_abs if v > 0]
        top_decreasing = [f"{k} ({v:.4f})" for k, v in sorted_by_abs if v < 0]

        return {
            "model_name": self.model_name,
            "experiment": self.experiment,
            "explainer_type": self.explainer_type,
            "predicted_risk_probability": round(pred_prob, 4),
            "base_value": round(base_val, 4),
            "features": self.features,
            "raw_feature_values": raw_vals_dict,
            "shap_attributions": shap_dict,
            "ranked_contributions": sorted_by_abs,
            "risk_increasing_factors": top_increasing,
            "risk_decreasing_factors": top_decreasing,
            "causality_disclaimer": self.CAUSALITY_DISCLAIMER,
        }

    def plot_waterfall(
        self,
        instance: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Renders an individual patient SHAP waterfall plot."""
        if output_path is None:
            output_path = FIGURES_DIR / "shap_waterfall.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        explanation = self.explain_instance(instance)
        sorted_contribs = sorted(explanation["shap_attributions"].items(), key=lambda x: x[1])

        features = [x[0] for x in sorted_contribs]
        vals = [x[1] for x in sorted_contribs]
        colors = ["#e74c3c" if v > 0 else "#2b5c8f" for v in vals]

        fig, ax = plt.subplots(figsize=(8, max(4.0, len(features) * 0.6)))
        bars = ax.barh(features, vals, color=colors, height=0.55)

        for bar in bars:
            width = bar.get_width()
            align = "left" if width >= 0 else "right"
            offset = 0.005 if width >= 0 else -0.005
            ax.annotate(
                f"{width:+.4f}",
                xy=(width + offset, bar.get_y() + bar.get_height() / 2),
                ha=align,
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("SHAP Value (Contribution to Log-Odds / Score)", fontsize=10)
        title_str = (
            f"Local Feature Attribution (SHAP) - Predicted Risk: "
            f"{explanation['predicted_risk_probability']:.1%}"
        )
        ax.set_title(title_str, fontsize=12, pad=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    def plot_global_importance(
        self,
        X_sample: Optional[pd.DataFrame] = None,
        output_path: Optional[Path] = None,
        max_display: int = 10,
    ) -> Path:
        """Computes global feature importance across evaluation cohort and renders bar plot."""
        if output_path is None:
            output_path = FIGURES_DIR / "shap_global_importance.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if X_sample is None:
            X_eval = self.background_sample
        else:
            X_eval = X_sample[self.features].copy()

        # Compute SHAP values across evaluation cohort
        if self.explainer_type in ("TreeExplainer", "LinearExplainer"):
            shap_obj = self.explainer(X_eval)
            vals = shap_obj.values
            if vals.ndim == 3 and vals.shape[2] == 2:
                vals = vals[:, :, 1]
        elif self.explainer_type == "KernelExplainer":
            k_shap = self.explainer.shap_values(X_eval)
            vals = k_shap[1] if isinstance(k_shap, list) else k_shap
        else:
            coefs = getattr(self.model, "coef_", np.ones((1, len(self.features))))[0]
            vals = np.abs(X_eval.values * coefs)

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)

        sorted_feats = [self.features[i] for i in sorted_idx][-max_display:]
        sorted_vals = mean_abs_shap[sorted_idx][-max_display:]

        fig, ax = plt.subplots(figsize=(8, max(4.5, len(sorted_feats) * 0.55)))
        bars = ax.barh(sorted_feats, sorted_vals, color="#2b5c8f", height=0.55)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.4f}",
                xy=(width + 0.005, bar.get_y() + bar.get_height() / 2),
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Mean Absolute SHAP Value: E[|SHAP value|]", fontsize=10)
        ax.set_title(
            f"Global Feature Importance ({self.model_name} - {len(self.features)} Features)",
            fontsize=12,
            pad=10,
        )
        ax.set_xlim(0, max(sorted_vals) * 1.25)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    @staticmethod
    def compare_with_expert_rules(
        shap_explanation: Dict[str, Any],
        expert_findings: List[str],
    ) -> Dict[str, Any]:
        """Provides a structured clinical comparison contrasting ML attributions with symbolic rules."""
        return {
            "statistical_ml_summary": {
                "predicted_risk_probability": shap_explanation["predicted_risk_probability"],
                "top_statistical_drivers": shap_explanation["risk_increasing_factors"][:3],
                "mitigating_statistical_factors": shap_explanation["risk_decreasing_factors"][:3],
                "paradigm": "Empirical statistical optimization (correlations across 3,392 training patients)",
            },
            "expert_system_summary": {
                "triggered_guideline_rules": expert_findings,
                "paradigm": "Deterministic medical guidelines (ACC/AHA, ATP III, ADA clinical thresholds)",
            },
            "synthesis_and_contrast": (
                "The statistical ML model assesses aggregate probability weight based on the training cohort, "
                "while the expert system identifies specific clinical threshold violations regardless of overall risk. "
                "Together, they provide both calibrated probability and actionable clinical justification."
            ),
        }


def main():
    """CLI execution demonstrating explainability pipeline."""
    print("=" * 70)
    print("HeartRiskAI: Model Explainability & SHAP Clinical Attribution")
    print("=" * 70)

    explainer = ClinicalModelExplainer()
    print(f"Loaded Model:     {explainer.model_name}")
    print(f"Features:         {explainer.features}")
    print(f"Explainer Type:   {explainer.explainer_type}")

    # Generate global importance plot
    global_path = explainer.plot_global_importance()
    print(f"\nGlobal feature importance saved to: {global_path}")

    # Explain sample patient
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    sample_patient = train_df[explainer.features].iloc[5].to_dict()
    print(f"\nExplaining sample patient instance:\n{sample_patient}")

    exp_result = explainer.explain_instance(sample_patient)
    print(f"Predicted Probability: {exp_result['predicted_risk_probability']:.1%}")
    print(f"Baseline Score:        {exp_result['base_value']:.4f}")
    print("Top Risk-Increasing Predictors:", exp_result["risk_increasing_factors"])
    print("Top Risk-Decreasing Predictors:", exp_result["risk_decreasing_factors"])

    waterfall_path = explainer.plot_waterfall(sample_patient)
    print(f"Waterfall plot saved to: {waterfall_path}")

    print("\n" + "=" * 70)
    print(explainer.CAUSALITY_DISCLAIMER)
    print("=" * 70)


if __name__ == "__main__":
    main()
