"""Reusable Data Auditing and Exploratory Data Analysis Module for HeartRiskAI.
Performs comprehensive data health checks, schema validation, missingness audits,
clinical plausibility checks, correlation analysis, and reports generation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    EXPECTED_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PHYSIOLOGICAL_BOUNDS,
    REPORTS_DIR,
    FIGURES_DIR,
)


class DataAuditException(Exception):
    """Custom exception raised when data auditing encounters fatal data anomalies."""
    pass


class DataAuditor:
    """Performs rigorous exploratory analysis and auditing on cardiovascular datasets."""

    def __init__(
        self,
        file_path: Path = RAW_DATA_PATH,
        target_col: str = TARGET_COLUMN,
        reports_dir: Path = REPORTS_DIR,
        figures_dir: Path = FIGURES_DIR,
    ):
        self.file_path = Path(file_path)
        self.target_col = target_col
        self.reports_dir = Path(reports_dir)
        self.figures_dir = Path(figures_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def inspect_file_status(self) -> Dict[str, Any]:
        """Inspects physical file existence and byte size before parsing."""
        if not self.file_path.exists():
            return {
                "exists": False,
                "size_bytes": 0,
                "is_empty": True,
                "status_message": f"File does not exist at path: {self.file_path}",
            }
        
        file_size = self.file_path.stat().st_size
        return {
            "exists": True,
            "size_bytes": file_size,
            "is_empty": (file_size == 0),
            "status_message": (
                f"File exists at {self.file_path} but is EMPTY (0 bytes)."
                if file_size == 0
                else f"File exists with size {file_size} bytes."
            ),
        }

    def load_dataset(self) -> pd.DataFrame:
        """Safely loads CSV dataset, validating file status and target column existence.
        
        Raises:
            DataAuditException: If file is missing, empty, or lacks expected target column.
        """
        file_status = self.inspect_file_status()
        if not file_status["exists"]:
            raise DataAuditException(
                f"[AUDIT FAILED] Dataset file is missing: {self.file_path}. "
                "Cannot proceed with EDA without authentic data."
            )
        if file_status["is_empty"]:
            raise DataAuditException(
                f"[AUDIT FAILED] Dataset file at {self.file_path} is completely empty (0 bytes). "
                "No columns, headers, or records exist. Expected target column "
                f"'{self.target_col}' is absent."
            )

        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            raise DataAuditException(f"Failed to parse CSV file: {e}") from e

        if df.empty or df.shape[0] == 0:
            raise DataAuditException(
                f"[AUDIT FAILED] CSV parsed 0 rows. Dataset is empty."
            )

        if self.target_col not in df.columns:
            raise DataAuditException(
                f"[AUDIT FAILED] Target column '{self.target_col}' not found in dataset columns: "
                f"{list(df.columns)}. Aborting audit to avoid fabricating results."
            )

        return df

    def run_full_audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs comprehensive statistical, structural, and clinical plausibility audits."""
        total_records, total_columns = df.shape
        columns_list = list(df.columns)
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # 1. Missingness analysis
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / total_records) * 100
        missing_summary = {
            col: {
                "count": int(missing_counts[col]),
                "percentage": round(float(missing_pcts[col]), 2),
            }
            for col in columns_list
        }
        complete_cases = int(df.dropna().shape[0])
        complete_case_pct = round((complete_cases / total_records) * 100, 2)

        # 2. Duplicate records
        duplicate_count = int(df.duplicated().sum())

        # 3. Target class analysis & class imbalance
        target_series = df[self.target_col].dropna()
        target_counts = target_series.value_counts().to_dict()
        target_proportions = (target_series.value_counts(normalize=True) * 100).to_dict()
        neg_count = int(target_counts.get(0, 0))
        pos_count = int(target_counts.get(1, 0))
        imbalance_ratio = round(neg_count / max(pos_count, 1), 2)

        # 4. Numerical feature distributions
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        num_stats = {}
        for c in num_cols:
            series = df[c].dropna()
            num_stats[c] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "q25": round(float(series.quantile(0.25)), 2),
                "median": round(float(series.median()), 2),
                "q75": round(float(series.quantile(0.75)), 2),
                "max": round(float(series.max()), 2),
                "skewness": round(float(series.skew()), 2),
            }

        # 5. Categorical feature distributions
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        cat_stats = {}
        for c in cat_cols:
            series = df[c].dropna()
            val_counts = series.value_counts().to_dict()
            cat_stats[c] = {
                "unique_values": int(series.nunique()),
                "frequencies": {str(k): int(v) for k, v in val_counts.items()},
                "mode": float(series.mode().iloc[0]) if not series.empty else None,
            }

        # 6. Physiological plausibility & clinical validation
        invalid_value_flags = {}
        for col, bounds in PHYSIOLOGICAL_BOUNDS.items():
            if col in df.columns:
                series = df[col].dropna()
                below_min = int((series < bounds["min"]).sum())
                above_max = int((series > bounds["max"]).sum())
                invalid_value_flags[col] = {
                    "expected_range": f"[{bounds['min']}, {bounds['max']}]",
                    "below_min_count": below_min,
                    "above_max_count": above_max,
                    "total_out_of_bounds": below_min + above_max,
                }

        # Specific behavioral contradictions
        clinical_logic_issues = {}
        if "currentSmoker" in df.columns and "cigsPerDay" in df.columns:
            non_smoker_with_cigs = int(
                ((df["currentSmoker"] == 0) & (df["cigsPerDay"] > 0)).sum()
            )
            smoker_with_zero_cigs = int(
                ((df["currentSmoker"] == 1) & (df["cigsPerDay"] == 0)).sum()
            )
            clinical_logic_issues["non_smoker_with_positive_cigs"] = non_smoker_with_cigs
            clinical_logic_issues["smoker_with_zero_cigs"] = smoker_with_zero_cigs

        # 7. Correlation analysis (Pearson with Target)
        numeric_and_target = [c for c in num_cols if c in df.columns]
        if self.target_col in df.columns:
            numeric_and_target.append(self.target_col)
        corr_matrix = df[numeric_and_target].corr()
        target_correlations = (
            corr_matrix[self.target_col].drop(self.target_col).to_dict()
            if self.target_col in corr_matrix
            else {}
        )

        return {
            "total_records": total_records,
            "total_columns": total_columns,
            "columns": columns_list,
            "dtypes": dtypes_dict,
            "target_column": self.target_col,
            "target_exists": (self.target_col in df.columns),
            "target_distribution": {
                "counts": {str(k): int(v) for k, v in target_counts.items()},
                "percentages": {str(k): round(float(v), 2) for k, v in target_proportions.items()},
                "imbalance_ratio": f"{imbalance_ratio}:1 (Negative:Positive)",
            },
            "missing_summary": missing_summary,
            "complete_cases": complete_cases,
            "complete_case_pct": complete_case_pct,
            "duplicate_count": duplicate_count,
            "numerical_summary": num_stats,
            "categorical_summary": cat_stats,
            "invalid_value_flags": invalid_value_flags,
            "clinical_logic_issues": clinical_logic_issues,
            "target_correlations": {
                k: round(float(v), 4) for k, v in target_correlations.items()
            },
            "records_available_for_analysis": {
                "raw_total": total_records,
                "complete_cases": complete_cases,
                "missing_any_field": total_records - complete_cases,
            },
        }

    def generate_visualizations(self, df: pd.DataFrame) -> List[Path]:
        """Generates academic-quality EDA visualizations and saves to reports/figures/."""
        saved_figures: List[Path] = []
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Target Class Distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        target_counts = df[self.target_col].value_counts().sort_index()
        colors = ["#2b5c8f", "#d95f02"]
        bars = ax.bar(["No CHD (0)", "10-Year CHD (1)"], target_counts.values, color=colors, width=0.5)
        for bar in bars:
            height = bar.get_height()
            pct = (height / len(df)) * 100
            ax.annotate(
                f"{height:,} ({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_title("Framingham Dataset: Target Class Distribution (TenYearCHD)", fontsize=12, pad=10)
        ax.set_ylabel("Patient Count")
        ax.set_ylim(0, max(target_counts.values) * 1.15)
        target_fig_path = self.figures_dir / "target_class_distribution.png"
        fig.tight_layout()
        fig.savefig(target_fig_path, dpi=300)
        plt.close(fig)
        saved_figures.append(target_fig_path)

        # 2. Missing Value Prevalence Bar Chart
        missing_counts = df.isnull().sum()
        missing_only = missing_counts[missing_counts > 0].sort_values(ascending=False)
        if not missing_only.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            pcts = (missing_only / len(df)) * 100
            y_pos = np.arange(len(missing_only))
            bars = ax.barh(y_pos, pcts.values, color="#e74c3c", height=0.55)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(missing_only.index)
            ax.invert_yaxis()
            for bar, count in zip(bars, missing_only.values):
                width = bar.get_width()
                ax.annotate(
                    f"{width:.1f}% ({count} rows)",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
            ax.set_xlabel("Percentage of Missing Values (%)")
            ax.set_title("Prevalence of Missing Clinical Biomarkers", fontsize=12, pad=10)
            ax.set_xlim(0, max(pcts.values) * 1.3)
            missing_fig_path = self.figures_dir / "missing_values_summary.png"
            fig.tight_layout()
            fig.savefig(missing_fig_path, dpi=300)
            plt.close(fig)
            saved_figures.append(missing_fig_path)

        # 3. Correlation Heatmap
        numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns] + [self.target_col]
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Pairwise Pearson Correlation Matrix (Physiological Features)", fontsize=12, pad=10)
        corr_fig_path = self.figures_dir / "correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(corr_fig_path, dpi=300)
        plt.close(fig)
        saved_figures.append(corr_fig_path)

        # 4. Numerical Feature Distributions (Histograms with KDE)
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        n_cols = len(num_cols)
        n_rows = int(np.ceil(n_cols / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows))
        axes = axes.flatten()
        for idx, col in enumerate(num_cols):
            ax = axes[idx]
            sns.histplot(
                data=df,
                x=col,
                hue=self.target_col,
                kde=True,
                palette=["#2b5c8f", "#d95f02"],
                ax=ax,
                bins=25,
                alpha=0.5,
            )
            ax.set_title(f"Distribution: {col}", fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        # Hide any unused axes
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])
        dist_fig_path = self.figures_dir / "numerical_feature_distributions.png"
        fig.tight_layout()
        fig.savefig(dist_fig_path, dpi=300)
        plt.close(fig)
        saved_figures.append(dist_fig_path)

        # 5. Key Risk Factor Boxplots by Target
        key_factors = ["age", "sysBP", "totChol", "glucose"]
        available_factors = [f for f in key_factors if f in df.columns]
        if available_factors:
            fig, axes = plt.subplots(1, len(available_factors), figsize=(3.5 * len(available_factors), 4.5))
            if len(available_factors) == 1:
                axes = [axes]
            for ax, col in zip(axes, available_factors):
                sns.boxplot(
                    data=df,
                    x=self.target_col,
                    y=col,
                    palette=["#a1c9f4", "#ffb482"],
                    ax=ax,
                    width=0.4,
                )
                ax.set_xticklabels(["No CHD", "CHD Event"])
                ax.set_title(f"{col} by Outcome", fontsize=11)
            box_fig_path = self.figures_dir / "key_biomarkers_boxplot.png"
            fig.tight_layout()
            fig.savefig(box_fig_path, dpi=300)
            plt.close(fig)
            saved_figures.append(box_fig_path)

        return saved_figures

    def generate_markdown_report(
        self, audit: Dict[str, Any], output_path: Optional[Path] = None
    ) -> Path:
        """Generates comprehensive eda_report.md summarizing audit findings."""
        if output_path is None:
            output_path = self.reports_dir / "eda_report.md"

        lines = [
            "# Exploratory Data Analysis & Cardiovascular Dataset Audit",
            "",
            "> **Project:** Heart Disease Risk Prediction (Genetic Algorithm + Machine Learning + Expert System)",
            f"> **Dataset Source File:** `{self.file_path.name}`",
            f"> **Target Clinical Endpoint:** `{self.target_col}` (10-Year Risk of Coronary Heart Disease)",
            "",
            "---",
            "",
            "## 1. Executive Summary & Record Accounting",
            "",
            f"- **Total Records Ingested:** {audit['total_records']:,}",
            f"- **Total Feature Columns:** {audit['total_columns']}",
            f"- **Target Column Present:** {'Yes' if audit['target_exists'] else 'NO'}",
            f"- **Complete Clinical Records (No Missingness):** {audit['complete_cases']:,} ({audit['complete_case_pct']}%)",
            f"- **Records with at Least One Missing Field:** {audit['records_available_for_analysis']['missing_any_field']:,}",
            f"- **Duplicate Rows:** {audit['duplicate_count']}",
            "",
            "---",
            "",
            "## 2. Target Class Distribution & Imbalance Risk",
            "",
            f"The dataset presents significant clinical class imbalance:",
            f"- **Class 0 (No CHD in 10 Years):** {audit['target_distribution']['counts'].get('0', 0):,} ({audit['target_distribution']['percentages'].get('0', 0.0)}%)",
            f"- **Class 1 (Developed CHD in 10 Years):** {audit['target_distribution']['counts'].get('1', 0):,} ({audit['target_distribution']['percentages'].get('1', 0.0)}%)",
            f"- **Observed Imbalance Ratio:** `{audit['target_distribution']['imbalance_ratio']}`",
            "",
            "> [!IMPORTANT]",
            "> **Evaluation Risk Protocol:**",
            "> Because negative instances predominate (~85%), raw accuracy is clinically deceptive.",
            "> In Phase 4, model evaluation must emphasize **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, and **Sensitivity**.",
            "> **Oversampling or SMOTE MUST NOT be performed prior to train-test splitting.**",
            "",
            "---",
            "",
            "## 3. Missing Value Prevalence Audit",
            "",
            "| Feature Column | Data Type | Missing Count | Missing Percentage | Imputation Strategy Planned |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for col, miss in audit["missing_summary"].items():
            dtype_str = audit["dtypes"].get(col, "unknown")
            count = miss["count"]
            pct = miss["percentage"]
            if count == 0:
                strategy = "None (Fully observed)"
            elif col in CATEGORICAL_FEATURES:
                strategy = "Mode / Most-frequent (Pipeline fit on train only)"
            else:
                strategy = "Median / Iterative (Pipeline fit on train only)"
            lines.append(f"| `{col}` | `{dtype_str}` | {count:,} | {pct}% | {strategy} |")

        lines.extend([
            "",
            "> [!NOTE]",
            "> **Data Leakage Safeguard:** Missing values must never be imputed across the entire dataset.",
            "> All imputers will be encapsulated inside `sklearn.pipeline.Pipeline` fitted solely on the training fold.",
            "",
            "---",
            "",
            "## 4. Numerical Feature Distributions",
            "",
            "| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max | Skewness |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for col, stats in audit["numerical_summary"].items():
            lines.append(
                f"| `{col}` | {stats['mean']} | {stats['std']} | {stats['min']} | "
                f"{stats['q25']} | {stats['median']} | {stats['q75']} | {stats['max']} | {stats['skewness']} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 5. Categorical Feature Distributions",
            "",
            "| Feature | Distinct Levels | Frequencies | Mode |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for col, stats in audit["categorical_summary"].items():
            freq_str = ", ".join([f"{k}: {v:,}" for k, v in stats["frequencies"].items()])
            lines.append(f"| `{col}` | {stats['unique_values']} | `{freq_str}` | {stats['mode']} |")

        lines.extend([
            "",
            "---",
            "",
            "## 6. Physiological Plausibility & Clinical Anomaly Screening",
            "",
            "Physiological validation bounds were audited against standard human clinical ranges:",
            "",
            "| Biomarker | Plausible Range | Out-of-Bounds Records | Clinical Note |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for col, flags in audit["invalid_value_flags"].items():
            lines.append(
                f"| `{col}` | `{flags['expected_range']}` | {flags['total_out_of_bounds']} | "
                f"{'Requires clinical capping/review' if flags['total_out_of_bounds'] > 0 else 'All observed values within physiological bounds'} |"
            )

        if audit["clinical_logic_issues"]:
            lines.extend([
                "",
                "**Behavioral Inconsistencies Detected:**",
                f"- Non-smoker (`currentSmoker == 0`) with `cigsPerDay > 0`: {audit['clinical_logic_issues'].get('non_smoker_with_positive_cigs', 0)} records",
                f"- Smoker (`currentSmoker == 1`) with `cigsPerDay == 0`: {audit['clinical_logic_issues'].get('smoker_with_zero_cigs', 0)} records",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 7. Correlation Analysis with 10-Year CHD Target",
            "",
            "Pearson correlations between numerical clinical predictors and the `TenYearCHD` target:",
            "",
            "| Numerical Predictor | Correlation (r) with `TenYearCHD` | Clinical Interpretation |",
            "| :--- | :--- | :--- |",
        ])

        sorted_corrs = sorted(
            audit["target_correlations"].items(), key=lambda x: abs(x[1]), reverse=True
        )
        for col, r in sorted_corrs:
            interpretation = (
                "Moderate positive association"
                if abs(r) > 0.15
                else "Mild positive association"
                if r > 0.05
                else "Weak association"
            )
            lines.append(f"| `{col}` | `{r:+.4f}` | {interpretation} |")

        lines.extend([
            "",
            "---",
            "",
            "## 8. Visual Artifacts Generated",
            "",
            "The following academic figures have been rendered to `reports/figures/`:",
            "1. `target_class_distribution.png`: Binary class distribution illustrating class imbalance.",
            "2. `missing_values_summary.png`: Prevalence of missing data across clinical biomarkers.",
            "3. `correlation_heatmap.png`: Full pairwise Pearson correlation matrix.",
            "4. `numerical_feature_distributions.png`: Multi-panel histograms with KDE by CHD outcome.",
            "5. `key_biomarkers_boxplot.png`: Distribution of age, blood pressure, cholesterol, and glucose by CHD outcome.",
            "",
            "---",
            "*(Report generated automatically by `src/data_audit.py`)*",
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path


def main():
    """CLI execution entrypoint for data audit."""
    print("=" * 70)
    print("HeartRiskAI: Cardiovascular Dataset Audit & Exploration")
    print("=" * 70)

    auditor = DataAuditor()
    status = auditor.inspect_file_status()

    print(f"Dataset path: {auditor.file_path}")
    print(f"File exists: {status['exists']}")
    print(f"File size: {status['size_bytes']} bytes")
    print(f"Status message: {status['status_message']}")

    if not status["exists"] or status["is_empty"]:
        print("\n" + "!" * 70)
        print("[AUDIT HALTED - DATASET MISSING OR EMPTY]")
        print(f"CRITICAL: The dataset file at '{auditor.file_path}' has a length of {status['size_bytes']} bytes.")
        print("There are no columns, no rows, and the target column 'TenYearCHD' is missing.")
        print("Per project protocol: Cannot invent model metrics or fabricate medical data.")
        print("!" * 70)
        sys.exit(1)

    print("\nLoading dataset and validating target column...")
    df = auditor.load_dataset()
    print(f"Successfully loaded dataset with shape: {df.shape}")

    print("\nRunning full clinical and statistical audit...")
    audit = auditor.run_full_audit(df)

    print(f"Total records: {audit['total_records']}")
    print(f"Features count: {audit['total_columns'] - 1} + 1 target ('{audit['target_column']}')")
    print(f"Complete cases: {audit['complete_cases']} ({audit['complete_case_pct']}%)")
    print(f"Target distribution: {audit['target_distribution']['counts']}")
    print(f"Imbalance ratio: {audit['target_distribution']['imbalance_ratio']}")

    print("\nGenerating academic visualizations...")
    figures = auditor.generate_visualizations(df)
    for fig_path in figures:
        print(f" - Saved: {fig_path.name}")

    print("\nCompiling markdown report...")
    report_path = auditor.generate_markdown_report(audit)
    print(f"Report written to: {report_path}")
    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()
