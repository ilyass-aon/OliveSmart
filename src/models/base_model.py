"""
OliveSmart — Base Model Interface
====================================
Abstract base class that defines the contract every model must fulfill.

All models in OliveSmart must:
  1. Implement fit() and predict()
  2. Expose optimize() for Optuna-based hyperparameter search
  3. Support save() / load() with joblib for reproducibility
  4. Report a consistent metrics dict via evaluate()
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.logger import get_logger

logger = get_logger("olivesmart")


# ============================================================
# Shared metric computation
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute the full set of regression metrics for OliveSmart.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    dict with keys:
        mae   : Mean Absolute Error
        rmse  : Root Mean Squared Error
        r2    : R² coefficient of determination
        mape  : Mean Absolute Percentage Error (%)
        n     : Number of samples evaluated
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE — guard against zero division
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4), "mape": round(mape, 4), "n": int(len(y_true))}


# ============================================================
# BaseModel
# ============================================================

class BaseModel(ABC):
    """
    Abstract base class for all OliveSmart models.

    Subclasses must implement:
      - fit(X_train, y_train)
      - predict(X)
      - optimize(X_train, y_train, tscv, n_trials) — may be a no-op

    Subclasses may override:
      - save(path) / load(path) — default uses joblib
      - get_hyperparams() — returns current hyperparameter dict
      - name — string identifier used in results

    Parameters
    ----------
    dataset_key : str
        Dataset this model is trained on (for logging and saving).
    random_seed : int
        Fixed seed for reproducibility.
    """

    def __init__(
        self,
        dataset_key: str = "FAOSTAT",
        random_seed: int = None,
    ):
        self.dataset_key = dataset_key
        self.random_seed = random_seed or config.OPTIMIZATION["random_seed"]
        self._is_fitted   = False
        self._train_time_s: float = 0.0
        self._best_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in reports and file names."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the given data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for input X."""

    @abstractmethod
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tscv,
        n_trials: int = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search.
        Returns the best hyperparameter dict found.
        For models with no search (Baseline), return {}.
        """

    # ------------------------------------------------------------------
    # Concrete methods (shared by all models)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        split_name: str = "test",
    ) -> Dict[str, float]:
        """
        Compute metrics for a given split.

        Parameters
        ----------
        X : np.ndarray
        y_true : np.ndarray
        split_name : str
            Label for logging ("train", "val", "test").

        Returns
        -------
        dict with mae, rmse, r2, mape, n — prefixed with split_name
        """
        if not self._is_fitted:
            raise RuntimeError(f"Model {self.name} must be fitted before evaluate().")

        y_pred = self.predict(X)
        metrics = compute_metrics(y_true, y_pred)

        logger.info(
            f"  [{self.name}] {split_name:5s} → "
            f"MAE={metrics['mae']:.1f} | RMSE={metrics['rmse']:.1f} | "
            f"R²={metrics['r2']:.4f} | MAPE={metrics['mape']:.2f}%"
        )
        return {f"{split_name}_{k}": v for k, v in metrics.items()}

    def evaluate_all_splits(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate on all three splits and return a combined results dict.

        Returns
        -------
        dict with all train_*/val_*/test_* metrics plus model metadata.
        """
        results = {}
        results.update(self.evaluate(X_train, y_train, "train"))
        results.update(self.evaluate(X_val,   y_val,   "val"))
        results.update(self.evaluate(X_test,  y_test,  "test"))
        results["model"]          = self.name
        results["dataset"]        = self.dataset_key
        results["train_time_s"]   = round(self._train_time_s, 3)
        results["best_params"]    = self._best_params.copy()
        results["random_seed"]    = self.random_seed
        return results

    def fit_timed(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Call fit() and record wall-clock training time."""
        t0 = time.time()
        self.fit(X_train, y_train)
        self._train_time_s = time.time() - t0
        self._is_fitted = True
        logger.info(f"  [{self.name}] Entraînement terminé en {self._train_time_s:.2f}s")

    def get_hyperparams(self) -> Dict[str, Any]:
        """Return the current (best) hyperparameters."""
        return self._best_params.copy()

    def save(self, out_dir: Optional[Path] = None) -> Path:
        """
        Save the model to disk using joblib.

        Parameters
        ----------
        out_dir : Path, optional
            Target directory. Defaults to config.MODELS_DIR / dataset_key.

        Returns
        -------
        Path to the saved file.
        """
        import joblib
        out_dir = out_dir or (config.MODELS_DIR / self.dataset_key)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.debug(f"Model saved: {path}")

        # Also save hyperparams as JSON for human readability
        meta_path = out_dir / f"{self.name}_params.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "model":       self.name,
                "dataset":     self.dataset_key,
                "best_params": self._best_params,
                "train_time_s": self._train_time_s,
                "random_seed": self.random_seed,
            }, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a saved model from disk."""
        import joblib
        model = joblib.load(path)
        logger.debug(f"Model loaded: {path}")
        return model
