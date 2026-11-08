"""Tests for classical regression models used in phase 4."""

import numpy as np
import pandas as pd
import pytest
from argparse import Namespace
from unittest.mock import Mock
from sklearn.model_selection import TimeSeriesSplit

import config
from main import run_pipeline
from src.data.preprocessing import Preprocessor
import src.models.classical_models as classical_models
from src.models.classical_models import build_classical_model


@pytest.fixture
def regression_data():
    X = pd.DataFrame({
        "year": np.arange(30, dtype=float),
        "area": np.linspace(10.0, 20.0, 30),
    })
    y = pd.Series(100.0 + 2.5 * X["year"] - 0.8 * X["area"])
    return X, y


def test_baseline_trains_and_predicts_mean(regression_data):
    X, y = regression_data
    model = build_classical_model("Baseline", "TEST")

    model.fit_timed(X.iloc[:20], y.iloc[:20])

    predictions = model.predict(X.iloc[20:])
    assert predictions.shape == (10,)
    assert np.allclose(predictions, y.iloc[:20].mean())


def test_ridge_search_uses_time_series_cv_and_evaluates(regression_data):
    X, y = regression_data
    splitter = Preprocessor(dataset_key="TEST")
    pipeline = splitter.build_pipeline("Ridge", X.iloc[:20])
    model = build_classical_model(
        "Ridge",
        "TEST",
        preprocessing_pipeline=pipeline,
    )

    best_params = model.optimize(
        X.iloc[:20],
        y.iloc[:20],
        TimeSeriesSplit(n_splits=3),
        n_trials=2,
    )
    model.fit_timed(X.iloc[:20], y.iloc[:20])

    metrics = model.evaluate(X.iloc[20:], y.iloc[20:], "test")
    assert "model__alpha" in best_params
    assert model.best_cv_mae is not None
    assert metrics["test_mae"] < 5


@pytest.mark.parametrize("model_name", ["RandomForest", "XGBoost"])
def test_tree_models_train_and_predict(model_name, regression_data):
    X, y = regression_data
    preprocessor = Preprocessor(dataset_key="TEST")
    pipeline = preprocessor.build_pipeline(model_name, X.iloc[:20])
    model = build_classical_model(
        model_name,
        "TEST",
        preprocessing_pipeline=pipeline,
    )

    model.fit_timed(X.iloc[:20], y.iloc[:20])

    assert model.predict(X.iloc[20:]).shape == (10,)


def test_unknown_phase_four_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported phase-4 model"):
        build_classical_model("MLP", "TEST")


def test_phase_four_pipeline_trains_classical_models(tmp_path, monkeypatch):
    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(config, "GRAPHS_DIR", tmp_path / "graphs")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setitem(config.REPORTING, "generate_graphs", False)
    monkeypatch.setitem(config.OPTIMIZATION, "randomized_n_iter", 1)

    run_pipeline(Namespace(
        dataset="FAOSTAT",
        model=None,
        force_rerun=False,
        phase=4,
    ))

    model_names = ["Baseline", "Ridge", "RandomForest", "XGBoost"]
    for model_name in model_names:
        result_path = (
            experiments_dir / "model_results" / f"FAOSTAT_{model_name}.json"
        )
        prediction_path = (
            experiments_dir / "predictions"
            / f"FAOSTAT_{model_name}_predictions.csv"
        )
        assert pd.read_json(result_path, typ="series")["status"] == "completed"
        assert prediction_path.is_file()
        assert (tmp_path / "models" / "FAOSTAT" / f"{model_name}.joblib").is_file()

    summary = pd.read_csv(experiments_dir / "phase4_model_comparison.csv")
    assert set(summary["model"]) == set(model_names)

    builder = Mock(side_effect=AssertionError(
        "Completed phase-4 models should be resumed from cache."
    ))
    monkeypatch.setattr(classical_models, "build_classical_model", builder)
    run_pipeline(Namespace(
        dataset="FAOSTAT",
        model=None,
        force_rerun=False,
        phase=4,
    ))
    builder.assert_not_called()
