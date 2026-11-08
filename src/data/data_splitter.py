"""
OliveSmart — Data Splitter Module
===================================
Responsible for splitting datasets into Train / Validation / Test partitions.

Scientific constraints enforced:
  - For time-series datasets: CHRONOLOGICAL split only (no shuffling)
  - Transformations are NEVER fitted on validation or test data
  - Split indices are saved for reproducibility and audit
  - Random seed is always fixed
  - Minimum sample thresholds are verified before splitting

Split strategies:
  - "chronological": strict time-ordered split (for FAOSTAT, NASA POWER, etc.)
  - "random":        random split with fixed seed (not recommended for time-series)
  - "timeseries_cv": TimeSeriesSplit for cross-validation (used during hyperparam search)

Usage:
    from src.data.data_splitter import DataSplitter

    splitter = DataSplitter()
    result = splitter.split(df, target_col="Yield_hg_ha", date_col="Year")

    X_train, y_train = result.X_train, result.y_train
    X_val,   y_val   = result.X_val,   result.y_val
    X_test,  y_test  = result.X_test,  result.y_test
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, NamedTuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.logger import get_logger

logger = get_logger("olivesmart")


# ============================================================
# SplitResult — structured output
# ============================================================

class SplitResult(NamedTuple):
    """
    Container for a Train / Validation / Test split.

    Attributes
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices for each partition.
    y_train, y_val, y_test : pd.Series
        Target vectors for each partition.
    train_idx, val_idx, test_idx : np.ndarray
        Original DataFrame indices for each partition (for audit).
    feature_cols : list of str
        Names of the feature columns used.
    target_col : str
        Name of the target column.
    split_meta : dict
        Metadata: strategy, ratios, n_samples, date_col, seed, etc.
    tscv : TimeSeriesSplit or None
        Configured TimeSeriesSplit for cross-validation on training data.
    """
    X_train: pd.DataFrame
    X_val:   pd.DataFrame
    X_test:  pd.DataFrame
    y_train: pd.Series
    y_val:   pd.Series
    y_test:  pd.Series
    train_idx: np.ndarray
    val_idx:   np.ndarray
    test_idx:  np.ndarray
    feature_cols: List[str]
    target_col:   str
    split_meta:   Dict[str, Any]
    tscv: Optional[TimeSeriesSplit]


# ============================================================
# DataSplitter — main class
# ============================================================

class DataSplitter:
    """
    Splits a DataFrame into Train / Validation / Test partitions.

    Parameters
    ----------
    train_ratio : float
        Fraction of data for training. Default from config.
    val_ratio : float
        Fraction for validation. Default from config.
    test_ratio : float
        Fraction for test. Default from config.
    random_seed : int
        Fixed seed for reproducibility.
    min_samples : int
        Minimum number of valid samples required to split.
    """

    def __init__(
        self,
        train_ratio:  float = None,
        val_ratio:    float = None,
        test_ratio:   float = None,
        random_seed:  int   = None,
        min_samples:  int   = None,
    ):
        cfg = config.SPLITTING
        self.train_ratio  = train_ratio  or cfg["train_ratio"]
        self.val_ratio    = val_ratio    or cfg["val_ratio"]
        self.test_ratio   = test_ratio   or cfg["test_ratio"]
        self.random_seed  = random_seed  if random_seed is not None else cfg["random_seed"]
        self.min_samples  = min_samples  or cfg["min_samples_for_split"]

        # Validate ratios
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 10)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total}. "
                f"({self.train_ratio} + {self.val_ratio} + {self.test_ratio})"
            )

    # ------------------------------------------------------------------
    # Main split entry point
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        date_col: Optional[str] = None,
        strategy: str = "chronological",
        dataset_key: Optional[str] = None,
    ) -> SplitResult:
        """
        Split a DataFrame into Train / Validation / Test.

        Parameters
        ----------
        df : pd.DataFrame
            The full prepared DataFrame (features + target).
        target_col : str
            Name of the target column (must exist in df).
        feature_cols : list of str, optional
            Explicit list of feature columns to use.
            If None, all numeric columns except target are used.
        date_col : str, optional
            Name of the time column. If provided, enables chronological sort.
        strategy : str
            "chronological" (default for time-series) or "random".
        dataset_key : str, optional
            For logging and metadata.

        Returns
        -------
        SplitResult
        """
        ds_label = dataset_key or "unknown"
        logger.info(f"Splitting {ds_label} | strategy={strategy} | "
                    f"ratios={self.train_ratio}/{self.val_ratio}/{self.test_ratio}")

        # --- Validation ---
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Remove rows where target is NaN
        df_clean = df.dropna(subset=[target_col]).copy()
        n = len(df_clean)
        logger.info(f"  Lignes valides (target non-null) : {n}/{len(df)}")

        if n < self.min_samples:
            raise ValueError(
                f"Dataset has only {n} valid samples (minimum required: "
                f"{self.min_samples}). Cannot split reliably.\n"
                f"Consider Leave-One-Out CV or report as exploratory only."
            )

        # --- Sort chronologically if date column is available ---
        if strategy == "chronological":
            if date_col and date_col in df_clean.columns:
                df_clean = df_clean.sort_values(date_col).reset_index(drop=False)
                # Preserve original index
                orig_index = df_clean["index"].values
                df_clean = df_clean.drop(columns=["index"])
            else:
                logger.warning(
                    f"  strategy='chronological' but date_col='{date_col}' "
                    "not found — using existing row order."
                )
                orig_index = df_clean.index.values
                df_clean = df_clean.reset_index(drop=True)
        else:
            # Random split
            rng = np.random.default_rng(self.random_seed)
            shuffled = rng.permutation(n)
            df_clean = df_clean.iloc[shuffled].reset_index(drop=False)
            orig_index = df_clean["index"].values
            df_clean = df_clean.drop(columns=["index"])

        # --- Determine feature columns ---
        feature_cols = self._resolve_features(df_clean, target_col, feature_cols)
        logger.info(f"  Features utilisees ({len(feature_cols)}) : {feature_cols}")
        logger.info(f"  Cible : {target_col}")

        # --- Compute split boundaries ---
        n_train = int(np.floor(n * self.train_ratio))
        n_val   = int(np.floor(n * self.val_ratio))
        n_test  = n - n_train - n_val

        # Ensure each set has at least 1 sample
        if n_train < 1 or n_val < 1 or n_test < 1:
            raise ValueError(
                f"Split produces empty partition: "
                f"train={n_train}, val={n_val}, test={n_test}. "
                f"Increase dataset size or reduce n_splits."
            )

        train_end = n_train
        val_end   = n_train + n_val

        # --- Create partitions ---
        X = df_clean[feature_cols]
        y = df_clean[target_col]

        X_train = X.iloc[:train_end].copy()
        X_val   = X.iloc[train_end:val_end].copy()
        X_test  = X.iloc[val_end:].copy()

        y_train = y.iloc[:train_end].copy()
        y_val   = y.iloc[train_end:val_end].copy()
        y_test  = y.iloc[val_end:].copy()

        train_idx = orig_index[:train_end]
        val_idx   = orig_index[train_end:val_end]
        test_idx  = orig_index[val_end:]

        # --- TimeSeriesSplit for CV during hyperparameter search ---
        cv_splits = config.OPTIMIZATION["cv_n_splits"]
        # Ensure we don't exceed what's possible with training data
        max_splits = max(2, n_train - 1)
        cv_splits  = min(cv_splits, max_splits)
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # --- Log summary ---
        logger.info(f"  Train : {len(X_train)} samples "
                    f"({y_train.min():.0f}–{y_train.max():.0f} {target_col})")
        logger.info(f"  Val   : {len(X_val)} samples "
                    f"({y_val.min():.0f}–{y_val.max():.0f} {target_col})")
        logger.info(f"  Test  : {len(X_test)} samples "
                    f"({y_test.min():.0f}–{y_test.max():.0f} {target_col})")
        logger.info(f"  CV    : TimeSeriesSplit(n_splits={cv_splits})")

        # --- Warn if val/test sets are very small ---
        if len(X_val) < 3:
            logger.warning(
                f"  Validation set has only {len(X_val)} samples. "
                "Metric estimates will be very noisy."
            )
        if len(X_test) < 3:
            logger.warning(
                f"  Test set has only {len(X_test)} samples. "
                "Final evaluation may be unreliable."
            )

        # --- Build metadata ---
        split_meta = {
            "dataset_key":   ds_label,
            "strategy":      strategy,
            "date_col":      date_col,
            "target_col":    target_col,
            "feature_cols":  feature_cols,
            "n_total":       n,
            "n_train":       int(len(X_train)),
            "n_val":         int(len(X_val)),
            "n_test":        int(len(X_test)),
            "train_ratio":   self.train_ratio,
            "val_ratio":     self.val_ratio,
            "test_ratio":    self.test_ratio,
            "random_seed":   self.random_seed,
            "cv_n_splits":   cv_splits,
            "train_idx":     train_idx.tolist(),
            "val_idx":       val_idx.tolist(),
            "test_idx":      test_idx.tolist(),
            # Date range info (if temporal)
            "train_period":  self._get_period(df_clean, date_col, slice(0, train_end)),
            "val_period":    self._get_period(df_clean, date_col, slice(train_end, val_end)),
            "test_period":   self._get_period(df_clean, date_col, slice(val_end, None)),
        }

        return SplitResult(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            feature_cols=feature_cols,
            target_col=target_col,
            split_meta=split_meta,
            tscv=tscv,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def split_from_bundle(self, bundle, feature_cols=None) -> SplitResult:
        """
        Convenience method: split directly from a DatasetBundle.

        Parameters
        ----------
        bundle : DatasetBundle
            Loaded dataset from DataLoader.
        feature_cols : list, optional
            Explicit feature columns. If None, auto-detected.

        Returns
        -------
        SplitResult
        """
        ds_cfg    = config.DATASETS[bundle.dataset_key]
        target_col = bundle.meta.get("detected_target") or ds_cfg.get("target_column")
        date_col   = ds_cfg.get("date_column")
        strategy   = "chronological" if date_col else "random"

        if not target_col:
            raise ValueError(
                f"No target column defined for {bundle.dataset_key}. "
                "Check config.DATASETS and ensure has_yield_target=True."
            )

        return self.split(
            df=bundle.df,
            target_col=target_col,
            feature_cols=feature_cols,
            date_col=date_col,
            strategy=strategy,
            dataset_key=bundle.dataset_key,
        )

    def save(self, result: SplitResult, out_dir: Optional[Path] = None):
        """
        Save split metadata and indices to JSON.

        Parameters
        ----------
        result : SplitResult
        out_dir : Path, optional
        """
        out_dir = out_dir or config.EXPERIMENTS_DIR / "splits"
        out_dir.mkdir(parents=True, exist_ok=True)
        key  = result.split_meta["dataset_key"]
        path = out_dir / f"{key}_split.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.split_meta, f, indent=2, ensure_ascii=False)
        logger.debug(f"Split metadata saved: {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]],
    ) -> List[str]:
        """
        Determine the final list of feature columns.

        Rules:
        - If feature_cols is explicitly given, validate and return them.
        - Otherwise, use all numeric columns except the target and
          known metadata/ID columns.
        """
        # Columns to always exclude (metadata, identifiers, non-informative)
        always_exclude = {
            target_col,
            "Area", "Item", "Unit", "Unit_Yield", "Unit_Area",
            "Unit_Production", "Source", "Flag", "YearMonth",
            "Date", "Region",    # group identifiers, not features
        }

        if feature_cols is not None:
            # Validate provided columns
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found in DataFrame: {missing}")
            excluded = [c for c in feature_cols if c == target_col]
            if excluded:
                raise ValueError("Target column cannot be in feature_cols.")
            return list(feature_cols)

        # Auto-detect: numeric columns only, excluding metadata
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        auto_features = [c for c in num_cols if c not in always_exclude]

        if not auto_features:
            raise ValueError(
                f"No numeric feature columns found. "
                f"All numeric columns: {num_cols}. "
                f"Excluded: {always_exclude}"
            )
        return auto_features

    @staticmethod
    def _get_period(
        df: pd.DataFrame,
        date_col: Optional[str],
        slc: slice,
    ) -> Optional[Dict[str, Any]]:
        """Extract the temporal range of a partition."""
        if not date_col or date_col not in df.columns:
            return None
        subset = df[date_col].iloc[slc]
        if subset.empty:
            return None
        return {"start": str(subset.iloc[0]), "end": str(subset.iloc[-1])}
