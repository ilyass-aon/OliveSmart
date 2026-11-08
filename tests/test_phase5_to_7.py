"""Tests for deep-learning experiments and phase-7 reporting."""

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from main import run_pipeline
from src.models.deep_models import (
    TorchLSTMRegressor,
    TorchMLPRegressor,
    make_sequences,
)
from src.reporting import generate_phase7_reports


@pytest.fixture
def deep_learning_config(monkeypatch):
    monkeypatch.setitem(config.MLP_CONFIG, "n_layers_range", (1, 1))
    monkeypatch.setitem(config.MLP_CONFIG, "units_choices", [8])
    monkeypatch.setitem(config.MLP_CONFIG, "dropout_range", (0.0, 0.0))
    monkeypatch.setitem(config.MLP_CONFIG, "lr_range", (1e-3, 1e-3))
    monkeypatch.setitem(config.MLP_CONFIG, "batch_size_choices", [8])
    monkeypatch.setitem(config.MLP_CONFIG, "l2_reg_range", (1e-5, 1e-5))
    monkeypatch.setitem(config.MLP_CONFIG, "max_epochs", 2)
    monkeypatch.setitem(config.MLP_CONFIG, "early_stopping_patience", 2)
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "hidden_sizes", [8])
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "n_layers_range", (1, 1))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "dropout_range", (0.0, 0.0))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "lr_range", (1e-3, 1e-3))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "batch_size_choices", [8])
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "l2_reg_range", (1e-5, 1e-5))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "max_epochs", 2)
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "early_stopping_patience", 2)
    monkeypatch.setitem(config.OPTIMIZATION, "max_time_per_model_s", 30)


def test_mlp_optimizer_comparison_and_final_fit(deep_learning_config):
    assert deep_learning_config is None
    assert config.MLP_CONFIG["max_epochs"] == 2
    values = np.linspace(-1, 1, 24)
    X = pd.DataFrame({"x1": values, "x2": values**2})
    y = pd.Series(1500 + 300 * values - 50 * values**2)
    pipeline = Pipeline([("scaler", StandardScaler())])

    for optimizer_name in ("adam", "adamw", "sgd"):
        model = TorchMLPRegressor(
            "TEST", optimizer_name, pipeline, random_seed=7
        )
        model.optimize(X.iloc[:20], y.iloc[:20], n_trials=1)
        model.fit_timed(X.iloc[:20], y.iloc[:20])
        predictions = model.predict(X.iloc[20:])
        assert model.name == f"MLP_{optimizer_name}"
        assert predictions.shape == (4,)
        assert np.isfinite(predictions).all()


def test_lstm_sequence_windows_and_fitting(deep_learning_config):
    assert deep_learning_config is None
    assert config.SEQUENCE_CONFIG["max_epochs"] == 2
    features = np.arange(60, dtype=np.float32).reshape(20, 3)
    sequences = make_sequences(features, lookback=3)
    validation = make_sequences(features[15:], lookback=3, context=features[13:15])
    assert sequences.shape == (18, 3, 3)
    assert validation.shape == (5, 3, 3)
    assert np.array_equal(validation[0], features[13:16])

    y = np.linspace(20, 40, len(sequences))
    model = TorchLSTMRegressor("TEST", optimizer_name="adam")
    model.optimize(sequences, y, n_trials=1)
    model.fit_timed(sequences, y)
    predictions = model.predict(validation)
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()


def test_make_sequences_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="Expected 2D"):
        make_sequences(np.array([1.0, 2.0]), lookback=2)


def test_phase7_creates_ranked_csv_graphs_and_reports(tmp_path, monkeypatch):
    results_dir = tmp_path / "model_results"
    predictions_dir = tmp_path / "predictions"
    reports_dir = tmp_path / "reports"
    graphs_dir = tmp_path / "graphs"
    results_dir.mkdir()
    predictions_dir.mkdir()
    monkeypatch.setitem(config.REPORTING, "generate_graphs", True)

    completed = {
        "dataset": "FAOSTAT",
        "model": "Ridge",
        "status": "completed",
        "val_mae": 120.0,
        "test_mae": 150.0,
        "test_rmse": 175.0,
        "test_r2": 0.6,
    }
    with (results_dir / "FAOSTAT_Ridge.json").open("w", encoding="utf-8") as file:
        json.dump(completed, file)
    with (results_dir / "FAOSTAT_MLP_adam.json").open("w", encoding="utf-8") as file:
        json.dump({
            "dataset": "FAOSTAT",
            "model": "MLP_adam",
            "status": "skipped",
            "reason": "Not enough samples",
        }, file)
    pd.DataFrame({
        "split": ["validation", "test", "test"],
        "actual": [100.0, 150.0, 200.0],
        "predicted": [110.0, 160.0, 190.0],
    }).to_csv(predictions_dir / "FAOSTAT_Ridge_predictions.csv", index=False)

    output = generate_phase7_reports(
        model_results_dir=results_dir,
        predictions_dir=predictions_dir,
        reports_dir=reports_dir,
        graphs_dir=graphs_dir,
        datasets=["FAOSTAT"],
    )

    comparison = pd.read_csv(output["comparison_csv"])
    assert comparison["model"].tolist() == ["Ridge"]
    assert (reports_dir / "FAOSTAT" / "rapport_FAOSTAT.pdf").is_file()
    assert (reports_dir / "OLIVESMART_GLOBAL_REPORT.pdf").is_file()
    assert (graphs_dir / "FAOSTAT" / "PHASE7_model_comparison.png").is_file()
    assert (graphs_dir / "FAOSTAT" / "PHASE7_test_predictions.png").is_file()


def _redirect_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "GRAPHS_DIR", tmp_path / "graphs")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setitem(config.REPORTING, "generate_graphs", False)


def test_phase_six_integrates_lstm_fallback(tmp_path, monkeypatch):
    _redirect_outputs(tmp_path, monkeypatch)
    monkeypatch.setitem(config.MODELS["TFT"], "optimizers_to_compare", ["adam"])
    monkeypatch.setitem(config.OPTIMIZATION, "optuna_n_trials", 1)
    monkeypatch.setitem(config.OPTIMIZATION, "max_time_per_model_s", 20)
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "hidden_sizes", [4])
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "n_layers_range", (1, 1))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "dropout_range", (0.0, 0.0))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "lr_range", (1e-3, 1e-3))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "batch_size_choices", [8])
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "l2_reg_range", (1e-5, 1e-5))
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "max_epochs", 2)
    monkeypatch.setitem(config.SEQUENCE_CONFIG, "early_stopping_patience", 2)

    run_pipeline(Namespace(
        dataset="FAOSTAT",
        model="TFT",
        force_rerun=False,
        phase=6,
    ))

    result_path = (
        tmp_path / "experiments" / "model_results"
        / "FAOSTAT_LSTM_fallback_adam.json"
    )
    with result_path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    assert result["status"] == "completed"
    assert result["requested_model"] == "TFT"
    assert result["trained_model"] == "LSTM"
    assert result["n_train_sequences"] > 0
    assert (tmp_path / "models" / "FAOSTAT" / "LSTM_fallback_adam.joblib").is_file()


def test_phase_five_trains_all_mlp_optimizers(
    tmp_path, monkeypatch, deep_learning_config
):
    assert deep_learning_config is None
    _redirect_outputs(tmp_path, monkeypatch)
    monkeypatch.setitem(config.MODELS["MLP"], "min_observations_required", 20)
    monkeypatch.setitem(config.OPTIMIZATION, "optuna_n_trials", 1)

    run_pipeline(Namespace(
        dataset="FAOSTAT",
        model="MLP",
        force_rerun=False,
        phase=5,
    ))

    for optimizer in ("adam", "adamw", "sgd"):
        result_path = (
            tmp_path / "experiments" / "model_results"
            / f"FAOSTAT_MLP_{optimizer}.json"
        )
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        assert result["status"] == "completed"
        assert result["optimizer"] == optimizer
        assert result["val_mae"] >= 0
        assert (
            tmp_path / "models" / "FAOSTAT"
            / f"MLP_{optimizer}.joblib"
        ).is_file()


def test_phase_seven_pipeline_records_mlp_skip_and_reports(tmp_path, monkeypatch):
    _redirect_outputs(tmp_path, monkeypatch)

    run_pipeline(Namespace(
        dataset="FAOSTAT",
        model="MLP",
        force_rerun=False,
        phase=7,
    ))

    for optimizer in ("adam", "adamw", "sgd"):
        result_path = (
            tmp_path / "experiments" / "model_results"
            / f"FAOSTAT_MLP_{optimizer}.json"
        )
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        assert result["status"] == "skipped"
        assert "minimum configuré" in result["reason"]

    assert (tmp_path / "reports" / "FAOSTAT" / "rapport_FAOSTAT.pdf").is_file()
    assert (tmp_path / "reports" / "OLIVESMART_GLOBAL_REPORT.pdf").is_file()
    empty_comparison = pd.read_csv(
        tmp_path / "reports" / "BEST_MODELS_COMPARISON.csv"
    )
    assert empty_comparison.empty
    assert "val_mae" in empty_comparison.columns
