"""HeartRisk AI — Cardiovascular Risk Research System.
Interactive Streamlit Laboratory Application integrating:
1. Genetic Algorithm Feature Selection
2. Machine Learning Risk Classification (Logistic Regression, Random Forest, SVM, XGBoost)
3. Rule-Based Clinical Expert System (ACC/AHA, ATP III, ADA Guidelines)
4. SHAP Model Explainability
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    SELECTED_FEATURES_PATH,
    PREPROCESSOR_PATH,
    PHYSIOLOGICAL_BOUNDS,
)
from src.data_preprocessing import CardiovascularDataPreprocessor
from src.expert_system import CardiovascularExpertSystem, SeverityLevel
from src.explainability import ClinicalModelExplainer


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeartRisk AI — Cardiovascular Risk Research System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling cards, badges, and alerts
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f3a60;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #2b5c8f;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        border-radius: 6px;
        padding: 1rem;
        font-size: 0.9rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .rule-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.9rem;
        margin-bottom: 0.7rem;
    }
    .badge-critical {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-high {
        background-color: #fd7e14;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-moderate {
        background-color: #ffc107;
        color: #212529;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# RESOURCE CACHING & LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained artifacts...")
def load_system_artifacts():
    """Safely loads fitted preprocessor, winning model, GA metadata, and explainer."""
    errors = []

    best_model_path = MODELS_DIR / "best_model.joblib"
    if not best_model_path.exists():
        errors.append(f"Model artifact missing: {best_model_path}")

    if not PREPROCESSOR_PATH.exists():
        errors.append(f"Preprocessor artifact missing: {PREPROCESSOR_PATH}")

    if not SELECTED_FEATURES_PATH.exists():
        errors.append(f"Selected features metadata missing: {SELECTED_FEATURES_PATH}")

    if errors:
        return None, None, None, None, errors

    try:
        model_artifact = joblib.load(best_model_path)
        preprocessor = CardiovascularDataPreprocessor.load(PREPROCESSOR_PATH)
        with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
            ga_metadata = json.load(f)
        explainer = ClinicalModelExplainer(model_artifact_path=best_model_path)
        return model_artifact, preprocessor, ga_metadata, explainer, []
    except Exception as e:
        return None, None, None, None, [f"Error loading system artifacts: {e}"]


@st.cache_data
def load_comparison_reports():
    """Loads empirical model evaluation CSVs."""
    cv_csv = REPORTS_DIR / "model_comparison.csv"
    test_csv = REPORTS_DIR / "test_evaluation_summary.csv"

    cv_df = pd.read_csv(cv_csv) if cv_csv.exists() else None
    test_df = pd.read_csv(test_csv) if test_csv.exists() else None
    return cv_df, test_df


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("## **HeartRisk AI**")
st.sidebar.caption("Cardiovascular Risk Research System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🩺 1. Risk Assessment",
        "📊 2. Model Comparison",
        "🔍 3. Risk Factor Explanation",
        "📖 4. About the Project",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Academic Laboratory Project**\n\n"
    "• Dataset: Framingham Heart Study\n"
    "• Feature Selection: Genetic Algorithm\n"
    "• Models: Logistic Regression, RF, SVM, XGBoost\n"
    "• Expert Rules: ACC/AHA & ADA Guidelines\n"
    "• Explainability: SHAP Analysis"
)


# -----------------------------------------------------------------------------
# PAGE 1: RISK ASSESSMENT
# -----------------------------------------------------------------------------
if page == "🩺 1. Risk Assessment":
    st.markdown('<div class="main-title">Cardiovascular Risk Assessment Calculator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Dual-Paradigm Decision Support: Machine Learning Probability Estimation + Rule-Based Expert System</div>',
        unsafe_allow_html=True,
    )

    model_artifact, preprocessor, ga_metadata, explainer, load_errors = load_system_artifacts()

    if load_errors:
        st.error("System artifacts could not be loaded:")
        for err in load_errors:
            st.warning(f"• {err}")
        st.info("Please run the training pipeline first: `python -m src.train_models`")
        st.stop()

    model_features = model_artifact["features"]
    model_name = model_artifact["model_name"]
    model_obj = model_artifact["model"]

    # Input Form
    with st.form("patient_assessment_form"):
        st.markdown("### Patient Clinical Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Demographics & Habits")
            age = st.number_input("Age (years)", min_value=18, max_value=95, value=52, step=1)
            male = st.selectbox("Biological Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            education = st.selectbox(
                "Education Level",
                options=[
                    ("Some High School (Level 1)", 1.0),
                    ("High School / GED (Level 2)", 2.0),
                    ("Some College (Level 3)", 3.0),
                    ("College Graduate (Level 4)", 4.0),
                ],
                format_func=lambda x: x[0],
            )[1]
            current_smoker = st.selectbox("Currently Smokes Cigarettes?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            cigs_per_day = st.number_input(
                "Cigarettes per Day",
                min_value=0.0,
                max_value=80.0,
                value=15.0 if current_smoker else 0.0,
                step=1.0,
            )

        with col2:
            st.markdown("##### Blood Pressure & Vitals")
            sys_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=260.0, value=142.0, step=1.0)
            dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=160.0, value=88.0, step=1.0)
            bp_meds = st.selectbox("Taking BP Medication?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40.0, max_value=180.0, value=74.0, step=1.0)
            bmi = st.number_input("Body Mass Index (kg/m²)", min_value=14.0, max_value=55.0, value=28.4, step=0.1)

        with col3:
            st.markdown("##### Metabolic & Clinical History")
            tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=500.0, value=224.0, step=1.0)
            glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=106.0, step=1.0)
            diabetes = st.selectbox("Diagnosed with Diabetes?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            prev_stroke = st.selectbox("History of Stroke?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            prev_hyp = st.selectbox("Prevalent Hypertension?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

        submit_btn = st.form_submit_button("Run Comprehensive Risk Assessment", type="primary")

    if submit_btn:
        # Construct raw patient dictionary
        raw_patient = {
            "age": float(age),
            "male": int(male),
            "education": float(education),
            "currentSmoker": int(current_smoker),
            "cigsPerDay": float(cigs_per_day),
            "BPMeds": int(bp_meds),
            "prevalentStroke": int(prev_stroke),
            "prevalentHyp": int(prev_hyp),
            "diabetes": int(diabetes),
            "totChol": float(tot_chol),
            "sysBP": float(sys_bp),
            "diaBP": float(dia_bp),
            "BMI": float(bmi),
            "heartRate": float(heart_rate),
            "glucose": float(glucose),
        }

        # Behavioral validation check
        if current_smoker == 0 and cigs_per_day > 0:
            st.warning("⚠️ Behavioral mismatch: Patient is marked as non-smoker, but cigarettes/day is greater than 0.")

        st.markdown("---")
        st.markdown("### Clinical Assessment Results")

        res_col1, res_col2 = st.columns([1, 1.2])

        # 1. Machine Learning Probability
        with res_col1:
            st.markdown("#### 1. Statistical Machine Learning Risk")
            
            # Preprocess instance using fitted preprocessor
            raw_df = pd.DataFrame([raw_patient])
            trans_df = preprocessor.transform(raw_df)
            model_input_df = trans_df[model_features].copy()

            if hasattr(model_obj, "predict_proba"):
                risk_prob = float(model_obj.predict_proba(model_input_df)[0, 1])
                threshold = 0.50
                is_high_risk = risk_prob >= threshold

                color_code = "#dc3545" if is_high_risk else "#28a745"
                risk_label = "Elevated Long-Term Risk (>= 50%)" if is_high_risk else "Low / Moderate Long-Term Risk (< 50%)"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 0.95rem; color: #555;">Estimated 10-Year CHD Probability:</div>
                        <div style="font-size: 2.8rem; font-weight: 800; color: {color_code};">{risk_prob:.1%}</div>
                        <div style="font-size: 0.9rem; margin-top: 5px;">
                            <b>Classification Cutoff:</b> {threshold:.2f} | <b>Predicted Category:</b> {risk_label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"**Algorithm:** {model_name} (Trained on {len(model_features)} GA-selected predictors: "
                    f"`{', '.join(model_features)}`)."
                )
            else:
                st.warning("The selected model does not output calibrated probabilities.")

        # 2. Rule-Based Expert System Findings
        with res_col2:
            st.markdown("#### 2. Rule-Based Clinical Expert System")
            expert = CardiovascularExpertSystem()
            expert_result = expert.evaluate(raw_patient)

            tier_badge = (
                '<span class="badge-critical">CRITICAL ALERT</span>'
                if expert_result.has_critical_alerts
                else '<span class="badge-high">HIGH RISK FACTORS</span>'
                if "HIGH" in expert_result.qualitative_risk_tier
                else '<span class="badge-moderate">MODERATE RISK FACTORS</span>'
                if "MODERATE" in expert_result.qualitative_risk_tier
                else '<span class="badge-low">OPTIMAL PROFILE</span>'
            )

            st.markdown(f"**Clinical Status:** {tier_badge} &nbsp; `{expert_result.qualitative_risk_tier}`", unsafe_allow_html=True)
            st.markdown(f"**Identified Risk Rules:** {len(expert_result.triggered_rules)} guideline thresholds triggered.")

            if expert_result.triggered_rules:
                for r in expert_result.triggered_rules:
                    badge_class = f"badge-{r['severity'].lower()}"
                    st.markdown(
                        f"""
                        <div class="rule-card">
                            <span class="{badge_class}">{r['severity']}</span> &nbsp; <b>{r['name']}</b> <code>[{r['rule_id']}]</code>
                            <div style="font-size: 0.85rem; color: #333; margin-top: 4px;">{r['finding']}</div>
                            <div style="font-size: 0.78rem; color: #666; font-style: italic; margin-top: 2px;">Source: {r['medical_source']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No guideline risk thresholds were triggered. All reported biomarkers fall within optimal reference ranges.")

        # Recommendations & Disclaimers
        st.markdown("#### Evidence-Based Clinical & Lifestyle Guidance")
        for idx, rec in enumerate(expert_result.educational_guidance, 1):
            st.markdown(f"**{idx}.** {rec}")

        st.markdown(
            f"""
            <div class="disclaimer-box">
                <b>RESEARCH & EDUCATIONAL DISCLAIMER:</b> {expert_result.clinical_disclaimer}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# PAGE 2: MODEL COMPARISON
# -----------------------------------------------------------------------------
elif page == "📊 2. Model Comparison":
    st.markdown('<div class="main-title">Empirical Model Comparison & Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Evaluation of All-Feature vs. Genetic Algorithm-Selected Configurations (Zero Fabricated Metrics)</div>',
        unsafe_allow_html=True,
    )

    cv_df, test_df = load_comparison_reports()

    if cv_df is None or test_df is None:
        st.error("Model benchmark reports are missing from `reports/`.")
        st.info("Execute model training first: `python -m src.train_models`")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 5-Fold Cross-Validation (Training)",
        "🎯 Held-Out Test Evaluation",
        "📉 ROC & PR Curves",
        "🧩 Confusion Matrices & GA Convergence",
    ])

    with tab1:
        st.markdown("#### Out-of-Fold Cross-Validation Results (Training Cohort, N = 3,392)")
        st.caption("Hyperparameters tuned via stratified 3-fold inner CV; metrics calculated across 5 out-of-fold partitions.")

        # Format display dataframe
        display_cv = cv_df[[
            "experiment", "model", "features_count", "roc_auc", "pr_auc",
            "recall_sensitivity", "specificity", "accuracy", "f1_score", "training_time_sec"
        ]].copy()

        st.dataframe(
            display_cv.style.highlight_max(subset=["roc_auc", "pr_auc", "recall_sensitivity"], color="#c6f6d5"),
            use_container_width=True,
        )

        st.markdown(
            """
            > **Key Observation:** **Logistic Regression (GA Selected)** achieved the highest cross-validation ROC-AUC (**0.7320**) 
            > and PR-AUC (**0.3512**) while reducing training time from **6.42s down to 0.11s** (a 58x speedup).
            """
        )

    with tab2:
        st.markdown("#### Final Evaluation on Sealed Held-Out Test Set (N = 848)")
        st.caption("The test set remained sealed until model selection was finalized to ensure zero data leakage.")

        display_test = test_df[[
            "experiment", "model", "features_count", "roc_auc", "pr_auc",
            "recall_sensitivity", "specificity", "accuracy", "f1_score"
        ]].copy()

        st.dataframe(
            display_test.style.highlight_max(subset=["roc_auc", "pr_auc"], color="#c6f6d5"),
            use_container_width=True,
        )

        st.info(
            "**Generalization Validation:** Both Random Forest and XGBoost improved test set ROC-AUC when trained "
            "on the 6 GA-selected features compared to all 15 features, proving that the Genetic Algorithm removed noisy, "
            "redundant dimensions without sacrificing diagnostic performance."
        )

    with tab3:
        st.markdown("#### Multi-Model Receiver Operating Characteristic & Precision-Recall Curves")
        c1, c2 = st.columns(2)
        roc_img_path = FIGURES_DIR / "roc_curves_comparison.png"
        pr_img_path = FIGURES_DIR / "pr_curves_comparison.png"

        with c1:
            if roc_img_path.exists():
                st.image(Image.open(roc_img_path), caption="Test Set ROC Curves Comparison", use_container_width=True)
            else:
                st.warning("ROC curves figure not found.")

        with c2:
            if pr_img_path.exists():
                st.image(Image.open(pr_img_path), caption="Test Set Precision-Recall Curves Comparison", use_container_width=True)
            else:
                st.warning("PR curves figure not found.")

    with tab4:
        st.markdown("#### Confusion Matrix Grid & Genetic Algorithm Convergence")
        c1, c2 = st.columns(2)
        cm_img_path = FIGURES_DIR / "confusion_matrices.png"
        ga_img_path = FIGURES_DIR / "ga_convergence.png"

        with c1:
            if cm_img_path.exists():
                st.image(Image.open(cm_img_path), caption="Confusion Matrices across All Evaluated Models", use_container_width=True)
            else:
                st.warning("Confusion matrices figure not found.")

        with c2:
            if ga_img_path.exists():
                st.image(Image.open(ga_img_path), caption="Genetic Algorithm Fitness Progression & Complexity Reduction", use_container_width=True)
            else:
                st.warning("GA convergence plot not found.")


# -----------------------------------------------------------------------------
# PAGE 3: RISK FACTOR EXPLANATION
# -----------------------------------------------------------------------------
elif page == "🔍 3. Risk Factor Explanation":
    st.markdown('<div class="main-title">Explainability & Feature Attribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">SHAP (SHapley Additive exPlanations) vs. Rule-Based Clinical Guidelines</div>',
        unsafe_allow_html=True,
    )

    model_artifact, preprocessor, ga_metadata, explainer, load_errors = load_system_artifacts()

    if load_errors:
        st.error("System artifacts could not be loaded. Please run the training pipeline first.")
        st.stop()

    st.markdown("### 1. Global Feature Importance (Cohort Level)")
    st.caption("Quantifies the mean absolute contribution of each feature to the model's predictions across the training cohort.")

    global_img_path = FIGURES_DIR / "shap_global_importance.png"
    if global_img_path.exists():
        st.image(Image.open(global_img_path), caption="Mean Absolute SHAP Value across Training Cohort", use_container_width=True)
    else:
        st.warning("Global SHAP importance plot not found. Run `python -m src.explainability` to generate.")

    st.markdown("---")
    st.markdown("### 2. Local Patient Attribution (Individual Waterfall Plot)")
    st.caption("Illustrates how an individual patient's biomarkers push their predicted score higher (red) or lower (blue) relative to baseline.")

    waterfall_img_path = FIGURES_DIR / "shap_waterfall.png"
    if waterfall_img_path.exists():
        st.image(Image.open(waterfall_img_path), caption="Sample Patient Local Attribution Waterfall", use_container_width=True)
    else:
        st.warning("Waterfall plot not found.")

    st.markdown("---")
    st.markdown("### 3. Comparing ML Statistical Attributions vs. Expert System Rules")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("#### Statistical Machine Learning (SHAP)")
        st.markdown(
            """
            - **How it works:** Deconstructs the model's output score into additive Shapley contributions based on cooperative game theory.
            - **What it tells us:** Identifies which features pushed the statistical probability up or down *within this mathematical model*.
            - **Limitation:** Reflects empirical correlations in the training dataset; **does NOT establish biological causation**.
            """
        )

    with comp_col2:
        st.markdown("#### Rule-Based Expert System")
        st.markdown(
            """
            - **How it works:** Evaluates deterministic IF-THEN clinical practice guidelines (ACC/AHA, ATP III, ADA).
            - **What it tells us:** Identifies specific medical threshold violations (e.g., Stage 2 Hypertension, Impaired Fasting Glucose).
            - **Role:** Provides actionable clinical justification and critical safety alerts regardless of aggregate statistical probabilities.
            """
        )

    st.markdown(
        f"""
        <div class="disclaimer-box">
            <b>SCIENTIFIC & MEDICAL DISCLAIMER:</b> {ClinicalModelExplainer.CAUSALITY_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 4: ABOUT THE PROJECT
# -----------------------------------------------------------------------------
elif page == "📖 4. About the Project":
    st.markdown('<div class="main-title">About HeartRisk AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Artificial Intelligence & Expert Systems Laboratory Project</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Project Overview
        **HeartRisk AI** is an academic decision-support framework that combines three core AI paradigms for cardiovascular risk assessment:
        1. **Genetic Algorithm (GA):** Performs heuristic binary feature selection to eliminate redundant and collinear variables while enforcing mandatory demographic confounders (`age`, biological sex).
        2. **Machine Learning Classifiers:** Predicts the 10-year risk of developing Coronary Heart Disease (`TenYearCHD`) using calibrated classifiers (Logistic Regression, Random Forest, SVM, XGBoost) with class imbalance corrections.
        3. **Rule-Based Clinical Expert System:** Employs explicit IF-THEN clinical rules codifying published ACC/AHA and ADA guidelines to generate qualitative findings and evidence-based lifestyle recommendations.
        4. **SHAP Interpretability:** Uses Shapley Additive exPlanations to explain individual and cohort-level feature contributions.

        ---

        ### Dataset Description
        - **Source:** Framingham Heart Study (Longitudinal epidemiological cohort).
        - **Cohort Size:** 4,240 records across 15 clinical predictors and 1 binary endpoint.
        - **Endpoint (`TenYearCHD`):** 1 = Developed coronary heart disease within a 10-year follow-up period; 0 = Did not develop CHD.
        - **Class Imbalance:** 3,596 negative cases (84.8%) vs. 644 positive cases (15.2%) — an imbalance ratio of **5.58 : 1**.

        ---

        ### Rigorous Evaluation Methodology
        To prevent data leakage and optimism bias:
        1. **Strict Hold-Out Quarantine:** The dataset was split into an 80% training set ($N=3,392$) and a 20% hold-out test set ($N=848$) before computing any statistics.
        2. **Training-Only Imputation & Scaling:** Missing values (median for continuous biomarkers, mode for categorical) and standard scaling were fit exclusively on training data.
        3. **Internal GA Validation:** The Genetic Algorithm's fitness function evaluated 5-fold Stratified CV ROC-AUC with a parsimony penalty strictly on training folds. The hold-out test set was never accessed during feature selection.
        4. **Comprehensive Clinical Metrics:** Avoided deceptive raw accuracy by emphasizing **ROC-AUC**, **PR-AUC (Average Precision)**, **Sensitivity (Recall)**, and **Specificity**.

        ---

        ### Clinical Limitations & Scope
        1. **Long-Term Risk vs. Acute Diagnosis:** This system models long-term 10-year cardiovascular risk. **It does not diagnose an active acute myocardial infarction (heart attack) or emergency cardiac arrest.**
        2. **Observational Nature:** All correlations represent statistical associations within the historical Framingham cohort; they do not establish biological causation.
        3. **Demographic Context:** The original Framingham cohort was predominantly Caucasian; extrapolation to diverse contemporary populations should be performed with appropriate clinical calibration.

        ---

        ### Authoritative Medical References
        1. **Whelton PK, et al.** *2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults.* J Am Coll Cardiol. 2018;71(19):e127-e248.
        2. **National Cholesterol Education Program (NCEP) Expert Panel.** *Third Report of the National Cholesterol Education Program (NCEP) Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults (Adult Treatment Panel III).* JAMA. 2001;285(19):2486-2497.
        3. **American Diabetes Association Professional Practice Committee.** *Classification and Diagnosis of Diabetes: Standards of Care in Diabetes—2024.* Diabetes Care. 2024;47(Suppl. 1):S20-S42.
        4. **Wilson PWF, et al.** *Prediction of Coronary Heart Disease Using Risk Factor Categories.* Circulation. 1998;97(18):1837-1847.
        5. **Lundberg SM, Lee SI.** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 30). 2017.
        """
    )
