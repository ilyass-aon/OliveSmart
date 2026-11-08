"""
OliveSmart — Data Explorer Module
===================================
Performs a full Exploratory Data Analysis (EDA) on a loaded DatasetBundle.

This module answers the following scientific questions for each dataset:
  1. What are the dimensions and structure of the data?
  2. Which columns are numeric vs categorical?
  3. Are there missing values, and how many?
  4. Are there duplicate rows?
  5. What are the statistical distributions?
  6. Are there outliers (IQR method)?
  7. What are the correlations between variables?
  8. Does the dataset contain a valid yield target variable?
  9. Is supervised regression scientifically feasible?
  10. What is the recommended modeling strategy?

Design decisions:
  - No data modification: EDA is read-only
  - Results are saved as JSON (machine-readable) and CSV (human-readable)
  - Graphs are generated per-dataset and saved to results/graphs/
  - Feasibility verdict is logged clearly with justification

Usage:
    from src.data.data_loader import DataLoader
    from src.data.data_explorer import DataExplorer

    loader = DataLoader()
    bundle = loader.load("FAOSTAT")

    explorer = DataExplorer()
    report = explorer.explore(bundle)
    print(report.feasibility_verdict)
"""

import json
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, NamedTuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.logger import get_logger
from src.data.data_loader import DatasetBundle

logger = get_logger("olivesmart")

# Suppress matplotlib/seaborn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":      config.REPORTING["graph_dpi"],
    "savefig.dpi":     config.REPORTING["graph_dpi"],
    "figure.facecolor":"white",
    "axes.facecolor":  "white",
})


# ============================================================
# EDAReport — structured output of the exploration
# ============================================================

class EDAReport(NamedTuple):
    """
    Structured result of a DataExplorer.explore() call.

    Attributes
    ----------
    dataset_key : str
    summary : dict
        Dimension info, dtypes, missing values, duplicates.
    stats : pd.DataFrame
        describe() output for numeric columns.
    missing_analysis : pd.DataFrame
        Per-column missing value count and percentage.
    outlier_analysis : pd.DataFrame
        Per-column outlier count using IQR method.
    correlation_matrix : pd.DataFrame
        Pearson correlation of numeric columns.
    feasibility_verdict : dict
        {
          "is_supervised_feasible": bool,
          "target_column": str or None,
          "n_samples": int,
          "reasoning": str,
          "recommendation": str,
        }
    graph_paths : list of str
        Absolute paths to saved graph PNG files.
    """
    dataset_key: str
    summary: Dict[str, Any]
    stats: pd.DataFrame
    missing_analysis: pd.DataFrame
    outlier_analysis: pd.DataFrame
    correlation_matrix: pd.DataFrame
    feasibility_verdict: Dict[str, Any]
    graph_paths: List[str]


# ============================================================
# DataExplorer — main class
# ============================================================

class DataExplorer:
    """
    Runs a complete, read-only EDA on a DatasetBundle.

    Parameters
    ----------
    graphs_dir : Path, optional
        Root directory for saving graphs.
        Defaults to config.GRAPHS_DIR / <dataset_key>.
    results_dir : Path, optional
        Directory for saving JSON/CSV EDA reports.
        Defaults to config.EXPERIMENTS_DIR.
    """

    # Minimum samples required to attempt supervised ML
    MIN_SAMPLES_SUPERVISED = 20

    # IQR multiplier for outlier detection (standard = 1.5)
    IQR_MULTIPLIER = 1.5

    def __init__(
        self,
        graphs_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.graphs_dir = graphs_dir or config.GRAPHS_DIR
        self.results_dir = results_dir or config.EXPERIMENTS_DIR

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def explore(self, bundle: DatasetBundle, save: bool = True) -> EDAReport:
        """
        Perform a full EDA on the given DatasetBundle.

        Parameters
        ----------
        bundle : DatasetBundle
            Loaded dataset from DataLoader.
        save : bool
            Whether to save graphs and JSON report to disk.

        Returns
        -------
        EDAReport
        """
        key = bundle.dataset_key
        df  = bundle.df

        logger.info(f"--- EDA : {key} ({df.shape[0]} rows x {df.shape[1]} cols) ---")

        # 1. Summary
        summary = self._compute_summary(df, bundle.meta)

        # 2. Descriptive statistics
        stats = self._compute_stats(df)

        # 3. Missing values
        missing_df = self._analyze_missing(df)

        # 4. Outliers
        outlier_df = self._analyze_outliers(df)

        # 5. Correlations
        corr_matrix = self._compute_correlations(df)

        # 6. Supervised ML feasibility
        verdict = self._assess_feasibility(bundle, df)

        # 7. Graphs (save to disk)
        graph_paths = []
        if save and config.REPORTING["generate_graphs"]:
            graph_paths = self._generate_graphs(key, df, verdict)

        # 8. Save EDA report
        if save:
            self._save_report(key, summary, stats, missing_df, outlier_df,
                              corr_matrix, verdict)

        # 9. Print readable summary to logger
        self._log_summary(key, summary, missing_df, outlier_df, verdict)

        return EDAReport(
            dataset_key=key,
            summary=summary,
            stats=stats,
            missing_analysis=missing_df,
            outlier_analysis=outlier_df,
            correlation_matrix=corr_matrix,
            feasibility_verdict=verdict,
            graph_paths=graph_paths,
        )

    def explore_all(self, bundles: Dict[str, DatasetBundle]) -> Dict[str, EDAReport]:
        """
        Run EDA on all datasets.

        Parameters
        ----------
        bundles : dict
            Mapping from dataset_key -> DatasetBundle (from DataLoader.load_all()).

        Returns
        -------
        dict
            Mapping from dataset_key -> EDAReport.
        """
        reports = {}
        for key, bundle in bundles.items():
            try:
                reports[key] = self.explore(bundle)
            except Exception as exc:
                logger.error(f"EDA failed for {key}: {exc}")
        return reports

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def _compute_summary(
        self, df: pd.DataFrame, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute basic structural summary of the DataFrame."""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        n_duplicates = int(df.duplicated().sum())

        return {
            "n_rows":          int(df.shape[0]),
            "n_cols":          int(df.shape[1]),
            "n_numeric_cols":  len(num_cols),
            "n_categorical_cols": len(cat_cols),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "n_duplicates":    n_duplicates,
            "has_duplicates":  n_duplicates > 0,
            "total_missing":   int(df.isnull().sum().sum()),
            "total_cells":     int(df.shape[0] * df.shape[1]),
            "missing_pct_overall": round(
                100 * df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1), 2
            ),
            "file_path":       meta.get("file_path", "unknown"),
            "detected_target": meta.get("detected_target"),
            "explored_at":     datetime.now().isoformat(),
        }

    def _compute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive statistics for all numeric columns."""
        num_df = df.select_dtypes(include=np.number)
        if num_df.empty:
            return pd.DataFrame()
        stats = num_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        stats["skewness"] = num_df.skew()
        stats["kurtosis"] = num_df.kurtosis()
        stats["cv_%"] = (num_df.std() / num_df.mean().replace(0, np.nan) * 100).round(2)
        return stats.round(4)

    def _analyze_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-column missing value statistics."""
        total = df.shape[0]
        missing_count  = df.isnull().sum()
        missing_pct    = (missing_count / total * 100).round(2)
        rows = []
        for col in df.columns:
            rows.append({
                "column":        col,
                "dtype":         str(df[col].dtype),
                "n_missing":     int(missing_count[col]),
                "pct_missing":   float(missing_pct[col]),
                "n_valid":       int(total - missing_count[col]),
                "pct_valid":     round(100 - float(missing_pct[col]), 2),
                "severity":      (
                    "CRITICAL" if missing_pct[col] > 50 else
                    "HIGH"     if missing_pct[col] > 20 else
                    "MEDIUM"   if missing_pct[col] > 5  else
                    "LOW"      if missing_pct[col] > 0  else
                    "NONE"
                ),
            })
        return pd.DataFrame(rows)

    def _analyze_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers per numeric column using the IQR method.
        A value is an outlier if it falls below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
        """
        num_df = df.select_dtypes(include=np.number)
        rows = []
        for col in num_df.columns:
            series = num_df[col].dropna()
            if series.empty:
                continue
            Q1  = series.quantile(0.25)
            Q3  = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.IQR_MULTIPLIER * IQR
            upper = Q3 + self.IQR_MULTIPLIER * IQR
            n_out = int(((series < lower) | (series > upper)).sum())
            rows.append({
                "column":       col,
                "Q1":           round(float(Q1), 4),
                "Q3":           round(float(Q3), 4),
                "IQR":          round(float(IQR), 4),
                "lower_fence":  round(float(lower), 4),
                "upper_fence":  round(float(upper), 4),
                "n_outliers":   n_out,
                "pct_outliers": round(100 * n_out / len(series), 2),
            })
        return pd.DataFrame(rows)

    def _compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pearson correlation matrix for numeric columns."""
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] < 2:
            return pd.DataFrame()
        return num_df.corr(method="pearson").round(4)

    def _assess_feasibility(
        self, bundle: DatasetBundle, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Determine whether supervised regression is scientifically feasible
        for this dataset.

        Rules:
        1. Dataset must have a valid target column (yield/rendement).
        2. Target column must be numeric and non-constant.
        3. Must have at least MIN_SAMPLES_SUPERVISED non-NaN target values.
        4. Config must declare has_yield_target = True.

        Returns
        -------
        dict with keys: is_supervised_feasible, target_column, n_samples,
                        n_features, reasoning, recommendation, warnings
        """
        key = bundle.dataset_key
        ds_cfg = config.DATASETS.get(key, {})
        config_has_target = ds_cfg.get("has_yield_target", False)
        config_target_col = ds_cfg.get("target_column")

        warnings_list = []
        reasoning_parts = []

        # --- Step 1: Does config declare a target? ---
        if not config_has_target:
            reasoning_parts.append(
                f"config.DATASETS['{key}']['has_yield_target'] = False. "
                "No yield target declared for this dataset."
            )
            return self._verdict(
                feasible=False,
                target_col=None,
                n_samples=len(df),
                n_features=0,
                reasoning=" ".join(reasoning_parts),
                recommendation=(
                    "Perform exploratory analysis only. "
                    "If you later merge this dataset with FAOSTAT yield data "
                    "on matching years/regions, supervised regression may become feasible."
                ),
                warnings=warnings_list,
            )

        # --- Step 2: Find the target column ---
        target_col = bundle.meta.get("detected_target") or config_target_col
        if target_col is None or target_col not in df.columns:
            reasoning_parts.append(
                f"Target column '{target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
            return self._verdict(
                feasible=False,
                target_col=target_col,
                n_samples=len(df),
                n_features=0,
                reasoning=" ".join(reasoning_parts),
                recommendation="Check column names and update config.py target_column.",
                warnings=warnings_list,
            )

        # --- Step 3: Target must be numeric ---
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            reasoning_parts.append(
                f"Target column '{target_col}' is not numeric "
                f"(dtype={df[target_col].dtype})."
            )
            return self._verdict(
                feasible=False,
                target_col=target_col,
                n_samples=len(df),
                n_features=0,
                reasoning=" ".join(reasoning_parts),
                recommendation="Convert target column to numeric type.",
                warnings=warnings_list,
            )

        # --- Step 4: Valid sample count ---
        target_series = df[target_col].dropna()
        n_valid = len(target_series)

        if n_valid < self.MIN_SAMPLES_SUPERVISED:
            reasoning_parts.append(
                f"Only {n_valid} non-null observations in target column "
                f"'{target_col}'. Minimum required: {self.MIN_SAMPLES_SUPERVISED}."
            )
            return self._verdict(
                feasible=False,
                target_col=target_col,
                n_samples=n_valid,
                n_features=0,
                reasoning=" ".join(reasoning_parts),
                recommendation=(
                    "Dataset is too small for reliable train/val/test split. "
                    "Consider Leave-One-Out CV or report as exploratory only."
                ),
                warnings=warnings_list,
            )

        # --- Step 5: Target must not be constant ---
        if target_series.nunique() <= 1:
            reasoning_parts.append(
                f"Target column '{target_col}' has only 1 unique value "
                "(constant). Cannot train a regression model."
            )
            return self._verdict(
                feasible=False,
                target_col=target_col,
                n_samples=n_valid,
                n_features=0,
                reasoning=" ".join(reasoning_parts),
                recommendation="Investigate why target is constant. Data may be corrupted.",
                warnings=warnings_list,
            )

        # --- Step 6: Count usable features ---
        num_df = df.select_dtypes(include=np.number)
        feature_cols = [c for c in num_df.columns if c != target_col]
        n_features = len(feature_cols)

        # Warnings for small datasets
        if n_valid < 50:
            warnings_list.append(
                f"Small dataset ({n_valid} samples). Results may be unreliable. "
                "Use cross-validation and report confidence intervals."
            )
        if n_features < 2:
            warnings_list.append(
                f"Only {n_features} feature column(s) available. "
                "Model performance may be limited."
            )

        # Build positive reasoning
        reasoning_parts.append(
            f"Target column '{target_col}' found with {n_valid} valid observations. "
            f"Target range: [{target_series.min():.1f}, {target_series.max():.1f}]. "
            f"Available features: {n_features}. "
            f"Supervised regression is scientifically feasible."
        )

        # Determine split strategy recommendation
        time_col = ds_cfg.get("date_column")
        if time_col and time_col in df.columns:
            split_strategy = "TimeSeriesSplit (chronological order preserved)"
        else:
            split_strategy = "Random split with stratification"

        return self._verdict(
            feasible=True,
            target_col=target_col,
            n_samples=n_valid,
            n_features=n_features,
            reasoning=" ".join(reasoning_parts),
            recommendation=(
                f"Recommended split: {split_strategy}. "
                f"Models to try: Baseline, Ridge, RandomForest, XGBoost"
                + (", MLP" if n_valid >= 50 else " (MLP not recommended: too few samples)")
                + "."
            ),
            warnings=warnings_list,
            feature_columns=feature_cols,
            target_range=[float(target_series.min()), float(target_series.max())],
            target_std=float(target_series.std()),
        )

    @staticmethod
    def _verdict(
        feasible: bool,
        target_col: Optional[str],
        n_samples: int,
        n_features: int,
        reasoning: str,
        recommendation: str,
        warnings: List[str],
        **extras,
    ) -> Dict[str, Any]:
        verdict = {
            "is_supervised_feasible": feasible,
            "target_column":          target_col,
            "n_samples":              n_samples,
            "n_features":             n_features,
            "reasoning":              reasoning,
            "recommendation":         recommendation,
            "warnings":               warnings,
        }
        verdict.update(extras)
        return verdict

    # ------------------------------------------------------------------
    # Graph generation
    # ------------------------------------------------------------------

    def _generate_graphs(
        self, key: str, df: pd.DataFrame, verdict: Dict[str, Any]
    ) -> List[str]:
        """Generate and save all EDA graphs for a dataset."""
        out_dir = self.graphs_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        num_df = df.select_dtypes(include=np.number)
        if num_df.empty:
            logger.warning(f"{key}: no numeric columns — skipping graphs.")
            return saved

        # 1. Missing values heatmap
        p = self._plot_missing(df, key, out_dir)
        if p:
            saved.append(p)

        # 2. Distribution histograms (numeric columns)
        p = self._plot_distributions(num_df, key, out_dir)
        if p:
            saved.append(p)

        # 3. Correlation heatmap
        p = self._plot_correlation(num_df, key, out_dir)
        if p:
            saved.append(p)

        # 4. Boxplots (outlier visualization)
        p = self._plot_boxplots(num_df, key, out_dir)
        if p:
            saved.append(p)

        # 5. Time-series plot for target (if feasible and temporal)
        if verdict.get("is_supervised_feasible") and verdict.get("target_column"):
            p = self._plot_target_series(df, key, verdict, out_dir)
            if p:
                saved.append(p)

        logger.info(f"[OK] {key}: {len(saved)} EDA graphs saved in {out_dir}")
        return [str(p) for p in saved]

    def _plot_missing(self, df: pd.DataFrame, key: str, out_dir: Path) -> Optional[Path]:
        """Bar chart of missing value percentages per column."""
        pct = df.isnull().mean() * 100
        pct = pct[pct > 0]
        if pct.empty:
            logger.debug(f"{key}: no missing values — skipping missing plot.")
            return None
        fig, ax = plt.subplots(figsize=(10, max(4, len(pct) * 0.5)))
        pct.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#e74c3c")
        ax.set_xlabel("% Valeurs manquantes")
        ax.set_title(f"{key} — Valeurs manquantes par colonne", fontweight="bold")
        ax.axvline(5,  color="orange", linestyle="--", label="Seuil 5%")
        ax.axvline(20, color="red",    linestyle="--", label="Seuil 20%")
        ax.legend()
        plt.tight_layout()
        path = out_dir / "01_missing_values.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def _plot_distributions(
        self, num_df: pd.DataFrame, key: str, out_dir: Path
    ) -> Optional[Path]:
        """Grid of histograms for all numeric columns."""
        cols = num_df.columns.tolist()
        n = len(cols)
        if n == 0:
            return None
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(cols):
            series = num_df[col].dropna()
            axes[i].hist(series, bins=min(30, max(5, len(series) // 5)),
                         color="#3498db", edgecolor="white", alpha=0.85)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("")
            axes[i].yaxis.set_major_formatter(mticker.ScalarFormatter())
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"{key} — Distributions des variables numériques",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        path = out_dir / "02_distributions.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def _plot_correlation(
        self, num_df: pd.DataFrame, key: str, out_dir: Path
    ) -> Optional[Path]:
        """Pearson correlation heatmap."""
        if num_df.shape[1] < 2:
            return None
        corr = num_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(max(8, corr.shape[1]), max(6, corr.shape[0])))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax
        )
        ax.set_title(f"{key} — Matrice de corrélation (Pearson)",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        path = out_dir / "03_correlation_matrix.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def _plot_boxplots(
        self, num_df: pd.DataFrame, key: str, out_dir: Path
    ) -> Optional[Path]:
        """Boxplot grid for outlier visualization."""
        cols = num_df.columns.tolist()
        n = len(cols)
        if n == 0:
            return None
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(cols):
            series = num_df[col].dropna()
            axes[i].boxplot(series, patch_artist=True,
                            boxprops=dict(facecolor="#2ecc71", alpha=0.7),
                            medianprops=dict(color="red", linewidth=2))
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xticks([])
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"{key} — Boxplots (détection valeurs aberrantes)",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        path = out_dir / "04_boxplots.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def _plot_target_series(
        self,
        df: pd.DataFrame,
        key: str,
        verdict: Dict[str, Any],
        out_dir: Path,
    ) -> Optional[Path]:
        """Time-series line plot for the target variable."""
        target_col = verdict.get("target_column")
        ds_cfg = config.DATASETS.get(key, {})
        date_col = ds_cfg.get("date_column")

        if target_col not in df.columns:
            return None

        # Find time axis
        x_col = None
        for candidate in [date_col, "Year", "year", "Date", "date", "YearMonth"]:
            if candidate and candidate in df.columns:
                x_col = candidate
                break

        if x_col is None:
            x_col = df.index

        fig, ax = plt.subplots(figsize=(12, 5))
        target_series = df[target_col]

        if isinstance(x_col, str):
            x_vals = df[x_col]
        else:
            x_vals = df.index

        ax.plot(x_vals, target_series, marker="o", linewidth=2,
                color="#2c3e50", markersize=5, label=target_col)
        ax.fill_between(x_vals, target_series, alpha=0.15, color="#2c3e50")

        # Trend line
        valid_mask = target_series.notna()
        if valid_mask.sum() > 2:
            try:
                x_num = np.arange(len(x_vals))[valid_mask]
                y_num = target_series[valid_mask].values
                z = np.polyfit(x_num, y_num, 1)
                p = np.poly1d(z)
                ax.plot(x_vals, p(np.arange(len(x_vals))),
                        "--", color="#e74c3c", linewidth=1.5, label="Tendance linéaire")
            except Exception:
                pass

        ax.set_xlabel(x_col if isinstance(x_col, str) else "Index")
        ax.set_ylabel(target_col)
        ds_name = config.DATASETS.get(key, {}).get("name", key)
        ax.set_title(f"{ds_name}\nÉvolution de la variable cible : {target_col}",
                     fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / "05_target_time_series.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Save & Log
    # ------------------------------------------------------------------

    def _save_report(
        self,
        key: str,
        summary: dict,
        stats: pd.DataFrame,
        missing_df: pd.DataFrame,
        outlier_df: pd.DataFrame,
        corr_matrix: pd.DataFrame,
        verdict: dict,
    ):
        """Save EDA results as JSON and CSV files."""
        out_dir = self.results_dir / "eda" / key
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        report_data = {
            "summary":             summary,
            "feasibility_verdict": verdict,
        }
        json_path = out_dir / f"{key}_eda_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        # CSV files
        if not stats.empty:
            stats.to_csv(out_dir / f"{key}_stats.csv", encoding="utf-8-sig")
        if not missing_df.empty:
            missing_df.to_csv(out_dir / f"{key}_missing.csv", index=False, encoding="utf-8-sig")
        if not outlier_df.empty:
            outlier_df.to_csv(out_dir / f"{key}_outliers.csv", index=False, encoding="utf-8-sig")
        if not corr_matrix.empty:
            corr_matrix.to_csv(out_dir / f"{key}_correlations.csv", encoding="utf-8-sig")

        logger.debug(f"EDA report saved: {out_dir}")

    def _log_summary(
        self,
        key: str,
        summary: dict,
        missing_df: pd.DataFrame,
        outlier_df: pd.DataFrame,
        verdict: dict,
    ):
        """Print a readable EDA summary to the logger."""
        sep = "=" * 55
        logger.info(sep)
        logger.info(f"EDA SUMMARY : {key}")
        logger.info(sep)
        logger.info(f"  Dimensions   : {summary['n_rows']} lignes x {summary['n_cols']} colonnes")
        logger.info(f"  Numeriques   : {summary['n_numeric_cols']} colonnes")
        logger.info(f"  Categoriques : {summary['n_categorical_cols']} colonnes")
        logger.info(f"  Doublons     : {summary['n_duplicates']}")
        logger.info(f"  Manquants    : {summary['total_missing']} cellules "
                    f"({summary['missing_pct_overall']}%)")

        # Missing highlights
        if not missing_df.empty:
            critical = missing_df[missing_df["severity"].isin(["CRITICAL", "HIGH"])]
            if not critical.empty:
                logger.warning(f"  Colonnes avec beaucoup de manquants :")
                for _, row in critical.iterrows():
                    logger.warning(f"    - {row['column']}: {row['pct_missing']}% manquants")

        # Outliers highlights
        if not outlier_df.empty:
            high_out = outlier_df[outlier_df["pct_outliers"] > 5]
            if not high_out.empty:
                logger.warning(f"  Colonnes avec outliers (>5%) :")
                for _, row in high_out.iterrows():
                    logger.warning(f"    - {row['column']}: {row['n_outliers']} outliers ({row['pct_outliers']}%)")

        # Feasibility verdict
        logger.info(sep)
        feasible = verdict["is_supervised_feasible"]
        label = "[FAISABLE]  Regression supervisee" if feasible else "[EXPLORATOIRE] Pas de regression supervisee"
        logger.info(f"  Verdict ML : {label}")
        logger.info(f"  Raison     : {verdict['reasoning'][:120]}...")
        logger.info(f"  Recomm.    : {verdict['recommendation'][:120]}...")
        for w in verdict.get("warnings", []):
            logger.warning(f"  [!] {w}")
        logger.info(sep)
