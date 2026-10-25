# Heart Disease Risk Prediction (HeartRiskAI)

**HeartRisk AI — Cardiovascular Risk Research System**  
An Artificial Intelligence and Expert Systems Laboratory project combining:
1. **Genetic Algorithm** for clinical feature selection
2. **Machine Learning Classifiers** (Logistic Regression, Random Forest, SVM, XGBoost) for 10-year coronary heart disease (`TenYearCHD`) risk prediction
3. **Rule-Based Clinical Expert System** codifying ACC/AHA cardiovascular guidelines
4. **SHAP (SHapley Additive exPlanations)** for transparent clinical explainability
5. **Interactive Streamlit Web Dashboard** (`app.py`) for live clinical decision support

---

## Preprocessing Pipeline Architecture

The preprocessing pipeline ([`src/data_preprocessing.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/data_preprocessing.py)) enforces a strict leakage-free protocol:

```text
               [ Full Framingham Dataset (4,240 rows) ]
                                  │
                                  ▼
      [ Stratified Train/Test Split (80% Train / 20% Test, seed=42) ]
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
[ X_train (3,392 rows) ]                        [ X_test (848 rows) ]
          │                                               │
          ▼ (Fit & Transform)                             ▼ (Transform Only)
[ Fitted ColumnTransformer Pipeline ] ─────────────► [ Sealed Test Matrix ]
  - Median Imputation (Numerical)                     (Untouched until final eval)
  - Mode Imputation (Categorical)
  - StandardScaler (Numerical)
  - Passthrough Binary/Ordinal
          │
          ▼
[ Class Weights / Optional Training SMOTE ]
```

### Preprocessing Design Decisions:
1. **Strict Target Isolation:** `TenYearCHD` is separated before any transformations. If `TenYearCHD` is ever detected in input features, `DataPreprocessingException` is raised immediately.
2. **Stratified Splitting:** 80/20 train/test split preserving the exact 5.58:1 class ratio (`~15.2%` positive CHD events in both folds).
3. **No Leakage Imputation:** Missing values (e.g., glucose, total cholesterol, BPMeds) are computed strictly on `X_train` using median (skewed numerical) and mode (categorical).
4. **Feature Encoding & Scaling:** Binary clinical indicators (`male`, `currentSmoker`, `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`) and ordinal `education` (1-4) are preserved directly to maintain 1-to-1 SHAP interpretability. Numerical biomarkers (`age`, `sysBP`, `diaBP`, `totChol`, `BMI`, `heartRate`, `glucose`, `cigsPerDay`) are standardized using `StandardScaler`.
5. **Class Imbalance Support:** Balanced inverse class weights ($w_0 = 0.59$, $w_1 = 3.29$, XGBoost scale_pos_weight = 5.59) computed strictly from `y_train`. Optional SMOTE is provided with programmatic safeguards prohibiting application outside training folds.
6. **Mandatory Confounders Retention:** Clinical epidemiological standards dictate that `age` and biological sex (`male`) must remain in the feature set across all primary experiments, regardless of feature selection algorithm outputs.

---

## Interactive Streamlit Application (`app.py`)

Launch the web application:
```bash
streamlit run app.py
```

### Application Pages:
1. **🩺 1. Risk Assessment:**
   - Interactive clinical vitals input with physiological boundary validation.
   - Real-time 10-year CHD probability from the trained, evaluated model.
   - Independent rule-based expert system findings with clinical severity badges and citations.
   - Critical red-flag alert system and evidence-based lifestyle recommendations.
2. **📊 2. Model Comparison:**
   - Out-of-fold Cross-Validation benchmark table from `reports/model_comparison.csv`.
   - Held-out test set performance from `reports/test_evaluation_summary.csv`.
   - Multi-model ROC and Precision-Recall curve comparisons.
   - Confusion matrix grid and Genetic Algorithm convergence plot.
3. **🔍 3. Risk Factor Explanation:**
   - Global cohort feature importance bar chart (mean absolute SHAP).
   - Local patient waterfall attribution plot illustrating risk-increasing vs. risk-decreasing factors.
   - Structured comparison contrasting statistical ML attributions with symbolic expert system rules.
4. **📖 4. About the Project:**
   - Comprehensive background on the Framingham study, Genetic Algorithm, ML models, evaluation methodology, clinical limitations, and medical references.

---

## Repository Structure

- [`app.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/app.py): Streamlit 4-page interactive dashboard.
- [`src/config.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/config.py): Project configuration, paths, seeds, clinical thresholds.
- [`src/data_audit.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/data_audit.py): Reusable data auditing and exploratory analysis engine.
- [`src/data_preprocessing.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/data_preprocessing.py): Leakage-free preprocessing pipeline.
- [`src/genetic_algorithm.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/genetic_algorithm.py): Binary GA feature selection engine.
- [`src/train_models.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/train_models.py): Multi-model tuning and training pipeline.
- [`src/evaluate_models.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/evaluate_models.py): Clinical evaluation metrics and visual curves.
- [`src/expert_system.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/expert_system.py): Rule-based clinical inference engine.
- [`src/explainability.py`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/src/explainability.py): SHAP interpretability and clinical attribution.
- [`notebooks/eda.ipynb`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/notebooks/eda.ipynb): Interactive exploratory data analysis notebook.
- [`tests/`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/tests/): Full Pytest suite (35 passing unit & integration tests).
- [`reports/model_comparison.csv`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/reports/model_comparison.csv): Empirical CV results.
- [`reports/test_evaluation_summary.csv`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/reports/test_evaluation_summary.csv): Held-out test results.
- [`reports/figures/`](file:///c:/Users/Aryan%20Rastogi/Downloads/HeartRiskAI/reports/figures/): High-resolution generated plots.

---

## Running the Pytest Test Suite

```bash
pytest -v
```
All **35 unit and integration tests** pass, ensuring end-to-end pipeline integrity, data leakage prevention, rule evaluation accuracy, and dashboard stability.
