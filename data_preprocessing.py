"""Data Preprocessing Pipeline for HeartRiskAI.
Implements leakage-free stratified splitting, training-only imputation,
feature scaling, Scikit-learn ColumnTransformer packaging, class weight calculation,
training-fold SMOTE support, and mandatory clinical feature enforcement.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    EXPECTED_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
    MANDATORY_FEATURES,
    RANDOM_STATE,
    TEST_SPLIT_SIZE,
)

logger = logging.getLogger(__name__)


class DataPreprocessingException(Exception):
    """Exception raised for violations in data preprocessing or leakage prevention."""
    pass


def enforce_mandatory_features(
    selected_features: List[str],
    mandatory_features: List[str] = MANDATORY_FEATURES,
) -> List[str]:
    """Ensures mandatory clinical covariates (age, sex) are retained across all experiments.
    
    Even if metaheuristic feature selection (e.g., Genetic Algorithm) attempts to omit
    age or sex, clinical epidemiology requires their retention as primary confounders.
    
    Args:
        selected_features: List of candidate feature names.
        mandatory_features: List of non-negotiable clinical predictors.
        
    Returns:
        List of feature names containing all mandatory features while preserving order.
    """
    feature_set = set(selected_features)
    final_list = list(selected_features)
    for m in mandatory_features:
        if m not in feature_set:
            logger.info("Enforcing mandatory clinical feature: %s", m)
            final_list.append(m)
            feature_set.add(m)
    return final_list


def compute_class_weights(y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Computes balanced class weights and XGBoost scale_pos_weight.
    
    Uses standard balanced inverse class frequencies:
    w_j = N / (k * N_j)
    
    Args:
        y: Binary target series (0 and 1).
        
    Returns:
        Dictionary with 'class_weight_dict', 'scale_pos_weight', and counts.
    """
    y_arr = np.asarray(y).ravel()
    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise DataPreprocessingException(f"Expected binary classification target, found: {classes}")

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_arr)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}

    neg_count = int(np.sum(y_arr == 0))
    pos_count = int(np.sum(y_arr == 1))
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    return {
        "class_weight_dict": class_weight_dict,
        "scale_pos_weight": scale_pos_weight,
        "neg_count": neg_count,
        "pos_count": pos_count,
        "imbalance_ratio": f"{scale_pos_weight:.2f}:1",
    }


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SPLIT_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Performs stratified train/test split, ensuring the test set remains isolated.
    
    Args:
        df: Raw DataFrame containing features and target.
        target_col: Name of target column ('TenYearCHD').
        test_size: Proportion for test split (default 0.20).
        random_state: Deterministic random seed.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if target_col not in df.columns:
        raise DataPreprocessingException(
            f"Target column '{target_col}' not found in dataframe columns: {list(df.columns)}"
        )

    # 1. Strictly isolate features from target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Verify no missing labels in target
    if y.isnull().any():
        raise DataPreprocessingException("Target column contains null values; cannot stratify.")

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    # 3. Assert stratification parity (within 1% tolerance)
    train_pos_pct = float(y_train.mean()) * 100
    test_pos_pct = float(y_test.mean()) * 100
    if abs(train_pos_pct - test_pos_pct) > 1.0:
        raise DataPreprocessingException(
            f"Stratification divergence: train={train_pos_pct:.2f}%, test={test_pos_pct:.2f}%"
        )

    return X_train, X_test, y_train, y_test


class CardiovascularDataPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer pipeline for Framingham clinical features.
    
    Preprocessing design decisions:
    1. Leakage Prevention: Transformers are fit ONLY on X_train.
    2. Numerical Imputation: Median imputation handles skewed biomarkers (glucose, totChol).
    3. Categorical Imputation: Most-frequent (mode) imputation preserves binary states.
    4. Scaling: Configurable StandardScaler (default) or RobustScaler for outliers.
    5. Categorical Encoding: Pass-through preserves binary 0/1 indicator semantics and
       ordinal education levels (1-4), ensuring 1-to-1 SHAP interpretability.
    6. Feature Name Preservation: Full feature name alignment across all transformations.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        impute_num_strategy: str = "median",
        impute_cat_strategy: str = "most_frequent",
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.scaler_type = scaler_type.lower()
        self.impute_num_strategy = impute_num_strategy
        self.impute_cat_strategy = impute_cat_strategy
        self.numerical_features = numerical_features or NUMERICAL_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES

        self.column_transformer_: Optional[ColumnTransformer] = None
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def _get_scaler(self) -> BaseEstimator:
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "none":
            return FunctionTransformer(func=None, validate=False)
        else:
            raise DataPreprocessingException(
                f"Unknown scaler_type: '{self.scaler_type}'. Supported: 'standard', 'robust', 'minmax', 'none'"
            )

    def _build_pipeline(self) -> ColumnTransformer:
        """Constructs Scikit-learn ColumnTransformer."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.impute_num_strategy)),
            ("scaler", self._get_scaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.impute_cat_strategy)),
            ("passthrough", FunctionTransformer(func=None, validate=False)),
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_features),
                ("cat", cat_pipeline, self.categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return transformer

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CardiovascularDataPreprocessor":
        """Fits imputer and scaler statistics exclusively on training features."""
        if not isinstance(X, pd.DataFrame):
            raise DataPreprocessingException("Input X must be a pandas DataFrame.")

        # Guard against target leakage
        if TARGET_COLUMN in X.columns:
            raise DataPreprocessingException(
                f"Data Leakage Violation: Target column '{TARGET_COLUMN}' detected in feature matrix X!"
            )

        # Validate that required features exist in X
        missing_num = set(self.numerical_features) - set(X.columns)
        missing_cat = set(self.categorical_features) - set(X.columns)
        if missing_num or missing_cat:
            raise DataPreprocessingException(
                f"Features missing from input DataFrame. Numerical: {missing_num}, Categorical: {missing_cat}"
            )

        self.feature_names_in_ = list(X.columns)
        self.column_transformer_ = self._build_pipeline()
        self.column_transformer_.fit(X)

        # Output feature ordering matches numerical + categorical ordering
        self.feature_names_out_ = list(self.numerical_features) + list(self.categorical_features)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted transformations to X without updating statistics."""
        if not self.is_fitted_ or self.column_transformer_ is None:
            raise DataPreprocessingException("Preprocessor must be fitted before transforming data.")
        if not isinstance(X, pd.DataFrame):
            raise DataPreprocessingException("Input X must be a pandas DataFrame.")
        if TARGET_COLUMN in X.columns:
            raise DataPreprocessingException(
                f"Data Leakage Violation: Target column '{TARGET_COLUMN}' detected in input X during transform!"
            )

        transformed_array = self.column_transformer_.transform(X)
        return pd.DataFrame(
            transformed_array,
            columns=self.feature_names_out_,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits on training data and returns transformed DataFrame."""
        return self.fit(X, y).transform(X)

    def save(self, file_path: Union[str, Path] = PREPROCESSOR_PATH) -> Path:
        """Persists fitted preprocessor object to disk."""
        if not self.is_fitted_:
            raise DataPreprocessingException("Cannot save unfitted preprocessor.")
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, file_path: Union[str, Path] = PREPROCESSOR_PATH) -> "CardiovascularDataPreprocessor":
        """Loads serialized preprocessor object."""
        p = Path(file_path)
        if not p.exists():
            raise DataPreprocessingException(f"Preprocessor file not found at: {p}")
        obj = joblib.load(p)
        if not isinstance(obj, CardiovascularDataPreprocessor):
            raise DataPreprocessingException(f"Loaded object is not a CardiovascularDataPreprocessor: {type(obj)}")
        return obj


def apply_training_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    sampling_strategy: Union[float, str] = "auto",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies SMOTE oversampling exclusively to the training set.
    
    CRITICAL: This function must NEVER be applied to test data or the full dataset
    prior to train-test splitting.
    
    Args:
        X_train: Preprocessed training features DataFrame.
        y_train: Training labels Series.
        random_state: Seed for reproducible synthetic sample generation.
        sampling_strategy: Desired ratio of minority to majority class.
        
    Returns:
        Tuple of (X_train_resampled, y_train_resampled) preserving DataFrame column names.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise DataPreprocessingException("X_train must be a pandas DataFrame to preserve feature names.")
    if TARGET_COLUMN in X_train.columns:
        raise DataPreprocessingException("Target column detected in X_train during SMOTE generation.")

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_train_resampled = pd.DataFrame(X_res, columns=X_train.columns)
    y_train_resampled = pd.Series(y_res, name=y_train.name)

    return X_train_resampled, y_train_resampled


def run_preprocessing_pipeline(
    raw_data_path: Path = RAW_DATA_PATH,
    test_size: float = TEST_SPLIT_SIZE,
    scaler_type: str = "standard",
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Executes the full preprocessing workflow with leakage-free safeguards.
    
    Workflow:
    1. Load raw dataset.
    2. Stratified train/test split (test set held out).
    3. Fit CardiovascularDataPreprocessor on X_train only.
    4. Transform X_train and X_test.
    5. Compute class weights and imbalance ratio from y_train only.
    6. Verify mandatory features (age, sex) are preserved.
    7. Optionally persist artifacts to models/ and data/processed/.
    
    Returns:
        Dictionary containing datasets, preprocessor, and pipeline metadata.
    """
    raw_path = Path(raw_data_path)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise DataPreprocessingException(f"Raw data file at {raw_path} is missing or empty.")

    df = pd.read_csv(raw_path)
    X_train_raw, X_test_raw, y_train, y_test = stratified_split(
        df, target_col=TARGET_COLUMN, test_size=test_size, random_state=RANDOM_STATE
    )

    # Fit preprocessor on X_train only
    preprocessor = CardiovascularDataPreprocessor(scaler_type=scaler_type)
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Calculate class weights strictly from y_train
    class_weight_info = compute_class_weights(y_train)

    # Enforce mandatory clinical features
    all_features = enforce_mandatory_features(preprocessor.feature_names_out_)

    if save_artifacts:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        preprocessor.save(PREPROCESSOR_PATH)

        train_export = X_train_trans.copy()
        train_export[TARGET_COLUMN] = y_train.values
        train_export.to_csv(TRAIN_DATA_PATH, index=False)

        test_export = X_test_trans.copy()
        test_export[TARGET_COLUMN] = y_test.values
        test_export.to_csv(TEST_DATA_PATH, index=False)

    return {
        "preprocessor": preprocessor,
        "X_train": X_train_trans,
        "X_test": X_test_trans,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": all_features,
        "class_weight_info": class_weight_info,
        "train_shape": X_train_trans.shape,
        "test_shape": X_test_trans.shape,
    }


def main():
    """CLI execution for preprocessing pipeline."""
    print("=" * 70)
    print("HeartRiskAI: Leakage-Free Preprocessing Pipeline Execution")
    print("=" * 70)

    try:
        results = run_preprocessing_pipeline(save_artifacts=True)
        print(f"Stratified Train Split: {results['train_shape']} samples")
        print(f"Stratified Test Split:  {results['test_shape']} samples (Untouched hold-out)")
        print(f"Feature count:          {len(results['feature_names'])} features")
        print(f"Mandatory features:     {MANDATORY_FEATURES} preserved")
        print(f"Class Imbalance Ratio:  {results['class_weight_info']['imbalance_ratio']}")
        print(f"Class Weights (y_train):{results['class_weight_info']['class_weight_dict']}")
        print(f"Scale Pos Weight (XGB): {results['class_weight_info']['scale_pos_weight']:.2f}")
        print(f"Preprocessor artifact:  {PREPROCESSOR_PATH}")
        print(f"Cached train data:      {TRAIN_DATA_PATH}")
        print(f"Cached test data:       {TEST_DATA_PATH}")
        print("\nPreprocessing pipeline completed successfully.")
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
