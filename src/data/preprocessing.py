"""
OliveSmart — Preprocessing Module
====================================
Transforms raw feature matrices into model-ready arrays.

Key responsibilities:
  1. Feature selection — identify safe predictors (no data leakage)
  2. Missing value imputation
  3. Categorical encoding
  4. Scaling / standardization (model-dependent)
  5. Build sklearn Pipelines fitted ONLY on training data

Data leakage prevention rules (enforced):
  - All transformers (scalers, imputers) are fit ONLY on X_train
  - X_val and X_test are only transformed, never used to fit
  - Production_tonnes is flagged as a potential leakage variable for FAOSTAT
    (since Yield = Production / Area, it is not an independent predictor)
  - Lag features are computed before the split, using only past values

Usage:
    from src.data.preprocessing import Preprocessor

    prep = Preprocessor()

    # For Ridge or MLP (needs scaling):
    pipeline = prep.build_pipeline(model_type="Ridge", X_train=X_train)
    X_tr, X_v, X_te = prep.fit_transform_split(pipeline, X_train, X_val, X_test)

    # For RF or XGBoost (no scaling needed):
    pipeline = prep.build_pipeline(model_type="RandomForest", X_train=X_train)
    X_tr, X_v, X_te = prep.fit_transform_split(pipeline, X_train, X_val, X_test)
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.logger import get_logger

logger = get_logger("olivesmart")
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Custom transformer: Lag Feature Generator
# ============================================================

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Generates lag features for time-series data.

    For a time-ordered DataFrame, creates columns like:
        feature_lag1, feature_lag2, ...

    Important: lags are computed on the FULL sorted dataset BEFORE
    the train/val/test split, so they are always based on past values only.
    This transformer is used to annotate the DataFrame — it does not
    'fit' on training data since it's a deterministic operation.

    Parameters
    ----------
    columns : list of str
        Columns to lag.
    n_lags : int
        Number of lag periods to generate (default 1).
    """

    def __init__(self, columns: List[str], n_lags: int = 1):
        self.columns = columns
        self.n_lags  = n_lags

    def fit(self, X, y=None):
        return self   # Stateless

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue
            for lag in range(1, self.n_lags + 1):
                X[f"{col}_lag{lag}"] = X[col].shift(lag)
        return X


# ============================================================
# FAOSTAT-specific leakage detector
# ============================================================

# Columns that are mathematically derived from the target in FAOSTAT
# Yield_hg_ha = Production_tonnes * 10000 / Area_harvested_ha
# Using Production_tonnes as a predictor of Yield_hg_ha is circular.
FAOSTAT_LEAKAGE_COLS = {"Production_tonnes"}

# Columns that are safe predictors for FAOSTAT
FAOSTAT_SAFE_FEATURES = {"Area_harvested_ha", "Year"}


# ============================================================
# Preprocessor — main class
# ============================================================

class Preprocessor:
    """
    Builds model-specific preprocessing pipelines and applies them
    following the strict no-leakage protocol.

    Models that need scaling : Ridge, MLP  → StandardScaler
    Models that don't        : RandomForest, XGBoost, Baseline → passthrough

    Parameters
    ----------
    dataset_key : str, optional
        Enables dataset-specific logic (e.g., FAOSTAT leakage checks).
    """

    # Models requiring feature standardization
    SCALING_MODELS = {"Ridge", "MLP"}

    # Imputation strategy for missing numeric values
    IMPUTE_STRATEGY = "median"

    def __init__(self, dataset_key: Optional[str] = None):
        self.dataset_key = dataset_key
        self._fitted_pipelines: Dict[str, Pipeline] = {}

    # ------------------------------------------------------------------
    # Feature preparation (before split)
    # ------------------------------------------------------------------

    def prepare_faostat(
        self,
        df: pd.DataFrame,
        add_lags: bool = True,
        n_lags: int = 1,
    ) -> pd.DataFrame:
        """
        Prepare the FAOSTAT DataFrame for modeling.

        Steps:
          1. Detect and warn about collinear columns (Production_tonnes)
          2. Keep only safe predictors + target
          3. Optionally add lag features (previous year's yield)
          4. Drop rows with NaN introduced by lags

        Parameters
        ----------
        df : pd.DataFrame
            Loaded FAOSTAT wide DataFrame.
        add_lags : bool
            Whether to add lagged target as a feature (Yield_lag1).
        n_lags : int
            Number of lag periods.

        Returns
        -------
        pd.DataFrame
            Prepared DataFrame, sorted by Year, with lag features if requested.

        Scientific note:
            Yield_lag1 (previous year's yield) is a valid predictor because
            it was known at the time of prediction.
            Production_tonnes is EXCLUDED because it is derived from Yield.
        """
        logger.info("Preparation des features FAOSTAT...")

        # Check for leakage columns
        leakage_found = FAOSTAT_LEAKAGE_COLS.intersection(set(df.columns))
        if leakage_found:
            logger.warning(
                f"[LEAKAGE WARNING] Colonnes a risque detectees : {leakage_found}. "
                "Ces colonnes sont mathematiquement derivees de la cible "
                "(Production = Yield * Area / 10000). Elles seront exclues."
            )

        # Sort chronologically
        df = df.sort_values("Year").reset_index(drop=True)

        # Select safe columns
        target_col = "Yield_hg_ha"
        safe_cols  = [c for c in FAOSTAT_SAFE_FEATURES if c in df.columns]
        keep_cols  = safe_cols + [target_col]

        missing_target = [c for c in [target_col] if c not in df.columns]
        if missing_target:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        df_prep = df[keep_cols].copy()
        logger.info(f"  Features retenues : {safe_cols}")
        logger.info(f"  Cible             : {target_col}")
        logger.info(f"  Exclues (leakage) : {list(leakage_found)}")

        # Add lag features BEFORE split (uses full time series)
        if add_lags:
            lagger = LagFeatureTransformer(columns=[target_col], n_lags=n_lags)
            df_prep = lagger.transform(df_prep)
            lag_cols = [f"{target_col}_lag{i}" for i in range(1, n_lags + 1)]
            logger.info(f"  Features retardees ajoutees : {lag_cols}")

            # Drop rows with NaN from lags (the first n_lags rows)
            before = len(df_prep)
            df_prep = df_prep.dropna().reset_index(drop=True)
            after  = len(df_prep)
            if before > after:
                logger.info(f"  Lignes supprimees (lags NaN) : {before - after}")

        logger.info(f"  Dataset pret : {df_prep.shape[0]} lignes x {df_prep.shape[1]} colonnes")
        return df_prep

    def prepare_generic(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generic feature preparation for non-FAOSTAT datasets.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
        exclude_cols : list, optional
            Additional columns to exclude.

        Returns
        -------
        pd.DataFrame
        """
        always_exclude = {
            "Area", "Item", "Unit", "Unit_Yield", "Unit_Area", "Unit_Production",
            "Source", "Flag", "YearMonth", "Date",
        }
        if exclude_cols:
            always_exclude.update(exclude_cols)

        keep = [
            c for c in df.columns
            if c not in always_exclude or c == target_col
        ]
        return df[keep].copy()

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def build_pipeline(
        self,
        model_type: str,
        X_train: pd.DataFrame,
    ) -> Pipeline:
        """
        Build a scikit-learn preprocessing Pipeline appropriate for the model.

        Ridge, MLP → StandardScaler (sensitive to feature magnitude)
        RF, XGBoost, Baseline → No scaling (tree-based, scale-invariant)

        All pipelines include median imputation for safety.

        Parameters
        ----------
        model_type : str
            One of "Ridge", "MLP", "RandomForest", "XGBoost", "Baseline".
        X_train : pd.DataFrame
            Training features (used to identify numeric/categorical columns).

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline (call fit_transform_split() to apply).
        """
        num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

        needs_scaling = model_type in self.SCALING_MODELS

        # Numeric transformer
        if needs_scaling:
            num_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy=self.IMPUTE_STRATEGY)),
                ("scaler",  StandardScaler()),
            ])
            logger.debug(f"  Pipeline {model_type}: Imputer + StandardScaler")
        else:
            num_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy=self.IMPUTE_STRATEGY)),
                # No scaler — tree-based models are scale-invariant
            ])
            logger.debug(f"  Pipeline {model_type}: Imputer only (no scaling)")

        # Categorical transformer (ordinal encoding, safe for trees)
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
        ])

        # Combine
        transformers = []
        if num_cols:
            transformers.append(("num", num_transformer, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_transformer, cat_cols))

        col_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",   # Drop any remaining columns
        )

        pipeline = Pipeline([("preprocessor", col_transformer)])
        return pipeline

    # ------------------------------------------------------------------
    # Fit + Transform (no-leakage protocol)
    # ------------------------------------------------------------------

    def fit_transform_split(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_val:   pd.DataFrame,
        X_test:  pd.DataFrame,
        model_type: str = "unknown",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the pipeline on X_train ONLY, then transform all three splits.

        This is the core anti-leakage operation:
          - pipeline.fit_transform(X_train) — learns statistics from train only
          - pipeline.transform(X_val)       — applies learned stats to val
          - pipeline.transform(X_test)      — applies learned stats to test

        Parameters
        ----------
        pipeline : Pipeline
            Unfitted sklearn pipeline from build_pipeline().
        X_train, X_val, X_test : pd.DataFrame
            Feature matrices from DataSplitter.
        model_type : str
            For logging.

        Returns
        -------
        Tuple of np.ndarray : (X_train_t, X_val_t, X_test_t)
        """
        logger.info(f"  Fitting pipeline '{model_type}' on X_train ({X_train.shape})...")

        # FIT on training data only
        X_train_t = pipeline.fit_transform(X_train)

        # TRANSFORM val and test (never fit on them)
        X_val_t  = pipeline.transform(X_val)
        X_test_t = pipeline.transform(X_test)

        logger.info(f"  Shapes apres transformation : "
                    f"train={X_train_t.shape}, val={X_val_t.shape}, test={X_test_t.shape}")

        # Store for later use (e.g., inverse transform for predictions)
        self._fitted_pipelines[model_type] = pipeline

        return X_train_t, X_val_t, X_test_t

    def get_feature_names(self, pipeline: Pipeline, X_train: pd.DataFrame) -> List[str]:
        """
        Retrieve output feature names from a fitted pipeline.

        Parameters
        ----------
        pipeline : Pipeline
        X_train : pd.DataFrame

        Returns
        -------
        list of str
        """
        try:
            col_tf = pipeline.named_steps["preprocessor"]
            return list(col_tf.get_feature_names_out())
        except Exception:
            # Fallback: return positional names
            n_out = pipeline.transform(X_train[:1]).shape[1]
            return [f"feature_{i}" for i in range(n_out)]

    def save_pipeline_info(
        self,
        pipeline: Pipeline,
        model_type: str,
        X_train: pd.DataFrame,
        out_dir: Optional[Path] = None,
    ):
        """
        Save pipeline metadata to JSON (not the pipeline object itself —
        that is saved by joblib in the model saver).

        Parameters
        ----------
        pipeline : Pipeline
        model_type : str
        X_train : pd.DataFrame
        out_dir : Path, optional
        """
        out_dir = out_dir or config.EXPERIMENTS_DIR / "pipelines"
        out_dir.mkdir(parents=True, exist_ok=True)

        needs_scaling = model_type in self.SCALING_MODELS
        info = {
            "model_type":     model_type,
            "needs_scaling":  needs_scaling,
            "scaler":         "StandardScaler" if needs_scaling else "None",
            "imputer":        self.IMPUTE_STRATEGY,
            "input_features": list(X_train.columns),
            "n_input_features": int(X_train.shape[1]),
            "leakage_check":  "Production_tonnes excluded from FAOSTAT" if
                               self.dataset_key == "FAOSTAT" else "N/A",
        }
        path = out_dir / f"{self.dataset_key or 'dataset'}_{model_type}_pipeline.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.debug(f"Pipeline info saved: {path}")

    # ------------------------------------------------------------------
    # Leakage audit utility
    # ------------------------------------------------------------------

    def audit_leakage(
        self,
        df: pd.DataFrame,
        target_col: str,
        threshold: float = 0.99,
    ) -> List[str]:
        """
        Identify columns with suspiciously high correlation to the target.
        A correlation > threshold suggests potential data leakage.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
        threshold : float
            Absolute correlation threshold (default 0.99).

        Returns
        -------
        list of str
            Column names with |correlation| > threshold.
        """
        num_df = df.select_dtypes(include=np.number)
        if target_col not in num_df.columns:
            return []

        correlations = num_df.corrwith(num_df[target_col]).abs()
        suspicious   = correlations[
            (correlations > threshold) & (correlations.index != target_col)
        ].index.tolist()

        if suspicious:
            logger.warning(
                f"[LEAKAGE AUDIT] Colonnes avec correlation |r| > {threshold} "
                f"avec '{target_col}' : {suspicious}. "
                "Verifiez s'il s'agit d'une fuite de donnees."
            )
        else:
            logger.info(
                f"[LEAKAGE AUDIT] Aucune correlation suspecte trouvee "
                f"(seuil={threshold})."
            )

        return suspicious
