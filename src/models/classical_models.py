"""Classical regression models and hyperparameter search for phase 4."""

from typing import Any, Dict, Optional

import numpy as np
from joblib import parallel_backend
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

import config
from src.models.base_model import BaseModel


CLASSICAL_MODELS = {"Baseline", "Ridge", "RandomForest", "XGBoost"}


class ClassicalRegressor(BaseModel):
    """Train and evaluate a scikit-learn regression pipeline."""

    def __init__(
        self,
        dataset_key: str,
        model_name: str,
        estimator: Pipeline,
        param_distributions: Optional[Dict[str, list]] = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_key=dataset_key, random_seed=random_seed)
        self.model_name = model_name
        self.estimator = estimator
        self.param_distributions = param_distributions or {}
        self.best_cv_mae: Optional[float] = None

    @property
    def name(self) -> str:
        return self.model_name

    def fit(self, X_train: Any, y_train: Any) -> None:
        self.estimator.fit(X_train, y_train)

    def predict(self, X: Any) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"Model {self.name} must be fitted before predict().")
        return np.asarray(self.estimator.predict(X))

    def optimize(
        self,
        X_train: Any,
        y_train: Any,
        tscv: Any,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Select hyperparameters using training-only chronological CV."""
        if not self.param_distributions:
            self._best_params = {}
            return {}
        if tscv is None:
            raise ValueError(f"{self.name} requires a cross-validation splitter.")

        candidate_count = len(ParameterGrid(self.param_distributions))
        configured_trials = (
            n_trials
            if n_trials is not None
            else config.OPTIMIZATION["randomized_n_iter"]
        )
        if configured_trials < 1:
            raise ValueError("n_trials must be at least 1.")

        search = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.param_distributions,
            n_iter=min(configured_trials, candidate_count),
            scoring="neg_mean_absolute_error",
            cv=tscv,
            random_state=self.random_seed,
            n_jobs=config.OPTIMIZATION["n_jobs"],
            refit=False,
            error_score="raise",
        )
        with parallel_backend("threading"):
            search.fit(X_train, y_train)

        self._best_params = dict(search.best_params_)
        self.best_cv_mae = float(-search.best_score_)
        self.estimator.set_params(**self._best_params)
        return self._best_params.copy()


def build_classical_model(
    model_name: str,
    dataset_key: str,
    preprocessing_pipeline: Optional[Pipeline] = None,
    random_seed: Optional[int] = None,
) -> ClassicalRegressor:
    """Create one supported phase-4 model with its preprocessing pipeline."""
    seed = (
        random_seed
        if random_seed is not None
        else config.OPTIMIZATION["random_seed"]
    )

    if model_name not in CLASSICAL_MODELS:
        raise ValueError(
            f"Unsupported phase-4 model '{model_name}'. "
            f"Choose from {sorted(CLASSICAL_MODELS)}."
        )

    if model_name == "Baseline":
        estimator = Pipeline([("model", DummyRegressor(strategy="mean"))])
        return ClassicalRegressor(dataset_key, model_name, estimator, random_seed=seed)

    if preprocessing_pipeline is None:
        raise ValueError(f"{model_name} requires a preprocessing pipeline.")

    if model_name == "Ridge":
        model = Ridge()
        param_distributions = {
            "model__alpha": np.logspace(-4, 4, 30).tolist(),
            "model__fit_intercept": [True, False],
        }
    elif model_name == "RandomForest":
        model = RandomForestRegressor(random_state=seed, n_jobs=1)
        param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 3, 5],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt"],
        }
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for the XGBoost model. "
                "Install it with `python -m pip install xgboost`."
            ) from exc

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=seed,
            n_jobs=1,
        )
        param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [1, 5, 10],
        }
    estimator = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", model),
    ])
    return ClassicalRegressor(
        dataset_key=dataset_key,
        model_name=model_name,
        estimator=estimator,
        param_distributions=param_distributions,
        random_seed=seed,
    )
