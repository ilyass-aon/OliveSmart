"""PyTorch MLP and LSTM regressors for phases 5 and 6."""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from src.models.base_model import BaseModel


def _make_network(
    model_type: str,
    n_features: int,
    params: Dict[str, Any],
) -> nn.Module:
    if model_type == "mlp":
        layers = []
        input_size = n_features
        for _ in range(params["n_layers"]):
            layers.extend([
                nn.Linear(input_size, params["units"]),
                nn.ReLU(),
                nn.Dropout(params["dropout"]),
            ])
            input_size = params["units"]
        layers.append(nn.Linear(input_size, 1))
        return nn.Sequential(*layers)

    recurrent_layers = params["n_layers"]
    return nn.Sequential(
        _LSTMHead(
            n_features=n_features,
            hidden_size=params["hidden_size"],
            n_layers=recurrent_layers,
            dropout=params["dropout"],
        )
    )


class _LSTMHead(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(values)
        return self.output(sequence[:, -1, :])


def _new_optimizer(
    optimizer_name: str,
    parameters,
    learning_rate: float,
    weight_decay: float,
):
    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def _fit_network(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, Any],
    max_epochs: int,
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    patience: int = 0,
) -> Tuple[nn.Module, int, float]:
    """Train a network, optionally selecting its epoch on an inner holdout."""
    torch.manual_seed(int(params.get("seed", config.OPTIMIZATION["random_seed"])))
    np.random.seed(int(params.get("seed", config.OPTIMIZATION["random_seed"])))

    device = torch.device(
        "cuda" if config.SYSTEM["gpu_id"] != -1 and torch.cuda.is_available()
        else "cpu"
    )
    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_mean = float(y_train_array.mean())
    y_scale = float(y_train_array.std())
    if y_scale == 0:
        y_scale = 1.0
    y_train_scaled = (y_train_array - y_mean) / y_scale

    train_dataset = TensorDataset(
        X_train_tensor,
        torch.as_tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1),
    )
    generator = torch.Generator().manual_seed(
        int(params.get("seed", config.OPTIMIZATION["random_seed"]))
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(int(params["batch_size"]), len(train_dataset)),
        shuffle=True,
        generator=generator,
    )

    network = _make_network(
        model_type=model_type,
        n_features=int(X_train.shape[-1]),
        params=params,
    ).to(device)
    optimizer = _new_optimizer(
        params["optimizer"],
        network.parameters(),
        float(params["learning_rate"]),
        float(params.get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()

    evaluation_tensors = None
    if X_eval is not None and y_eval is not None:
        evaluation_tensors = (
            torch.as_tensor(X_eval, dtype=torch.float32, device=device),
            np.asarray(y_eval, dtype=np.float64).reshape(-1),
        )

    best_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(network.state_dict())
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        network.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = network(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

        if evaluation_tensors is None:
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
            continue

        network.eval()
        eval_X, eval_y = evaluation_tensors
        with torch.no_grad():
            scaled_predictions = network(eval_X).cpu().numpy().reshape(-1)
        predictions = scaled_predictions * y_scale + y_mean
        eval_mae = float(np.mean(np.abs(eval_y - predictions)))
        if eval_mae < best_loss:
            best_loss = eval_mae
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                break

    network.load_state_dict(best_state)
    network.to("cpu")
    network.eval()
    network._olivesmart_y_mean = y_mean
    network._olivesmart_y_scale = y_scale
    return network, best_epoch, best_loss


class TorchMLPRegressor(BaseModel):
    """MLP using Optuna trials on a chronological holdout from training data."""

    def __init__(
        self,
        dataset_key: str,
        optimizer_name: str,
        preprocessing_pipeline: Pipeline,
        random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_key, random_seed)
        self.optimizer_name = optimizer_name
        self.preprocessing_pipeline = preprocessing_pipeline
        self._fitted_preprocessor: Optional[Pipeline] = None
        self.network: Optional[nn.Module] = None
        self._target_mean = 0.0
        self._target_scale = 1.0
        self.best_inner_mae: Optional[float] = None
        self._best_epoch = 1

    @property
    def name(self) -> str:
        return f"MLP_{self.optimizer_name}"

    def _transform(self, X: Any, fit: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if fit:
                self._fitted_preprocessor = clone(self.preprocessing_pipeline)
                values = self._fitted_preprocessor.fit_transform(X)
            else:
                if self._fitted_preprocessor is None:
                    raise RuntimeError("MLP preprocessing has not been fitted.")
                values = self._fitted_preprocessor.transform(X)
        else:
            values = X
        return np.asarray(values, dtype=np.float32)

    def optimize(
        self,
        X_train: Any,
        y_train: Any,
        tscv: Any = None,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        if len(X_train) < 8:
            raise ValueError("MLP tuning requires at least 8 training rows.")

        boundary = min(max(int(len(X_train) * 0.8), 2), len(X_train) - 1)
        X_fit_raw = X_train.iloc[:boundary] if isinstance(X_train, pd.DataFrame) else X_train[:boundary]
        X_eval_raw = X_train.iloc[boundary:] if isinstance(X_train, pd.DataFrame) else X_train[boundary:]
        y_values = np.asarray(y_train, dtype=np.float32).reshape(-1)
        X_fit = self._transform(X_fit_raw, fit=True)
        X_eval = self._transform(X_eval_raw)
        y_fit, y_eval = y_values[:boundary], y_values[boundary:]
        mlp_config = config.MLP_CONFIG
        max_epochs = int(mlp_config["max_epochs"])
        patience = int(mlp_config["early_stopping_patience"])
        budget = config.OPTIMIZATION["max_time_per_model_s"]
        n_trials = n_trials or config.OPTIMIZATION["optuna_n_trials"]
        def objective(trial: optuna.Trial) -> float:
            n_layers = trial.suggest_int(
                "n_layers", *mlp_config["n_layers_range"]
            )
            params = {
                "n_layers": n_layers,
                "units": trial.suggest_categorical(
                    "units", mlp_config["units_choices"]
                ),
                "dropout": trial.suggest_float(
                    "dropout", *mlp_config["dropout_range"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *mlp_config["lr_range"], log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", mlp_config["batch_size_choices"]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", *mlp_config["l2_reg_range"], log=True
                ),
                "optimizer": self.optimizer_name,
                "seed": self.random_seed + trial.number,
            }
            _, best_epoch, inner_mae = _fit_network(
                "mlp",
                X_fit,
                y_fit,
                params,
                max_epochs,
                X_eval=X_eval,
                y_eval=y_eval,
                patience=patience,
            )
            trial.set_user_attr("best_epoch", best_epoch)
            return inner_mae

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        per_optimizer_budget = (
            budget / len(config.MODELS["MLP"]["optimizers_to_compare"])
            if budget is not None else None
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=per_optimizer_budget,
            show_progress_bar=False,
        )
        if not study.trials:
            raise RuntimeError(f"Optuna did not run any trial for {self.name}.")

        best_trial = study.best_trial
        self._best_epoch = int(best_trial.user_attrs.get("best_epoch", max_epochs))
        self.best_inner_mae = float(best_trial.value)
        self._best_params = dict(best_trial.params)
        self._best_params["optimizer"] = self.optimizer_name
        self._best_params["selected_epochs"] = self._best_epoch
        self._best_params["inner_train_mae"] = self.best_inner_mae
        return self.get_hyperparams()

    def fit(self, X_train: Any, y_train: Any) -> None:
        X_values = self._transform(X_train, fit=True)
        params = dict(self._best_params)
        params["seed"] = self.random_seed
        params["optimizer"] = self.optimizer_name
        self.network, _, _ = _fit_network(
            "mlp",
            X_values,
            np.asarray(y_train),
            params,
            max_epochs=self._best_epoch,
        )
        self._target_mean = self.network._olivesmart_y_mean
        self._target_scale = self.network._olivesmart_y_scale

    def predict(self, X: Any) -> np.ndarray:
        if not self._is_fitted or self.network is None:
            raise RuntimeError(f"Model {self.name} must be fitted before predict().")
        X_values = torch.as_tensor(self._transform(X), dtype=torch.float32)
        with torch.no_grad():
            predictions = self.network(X_values).numpy().reshape(-1)
        return predictions * self._target_scale + self._target_mean


class TorchLSTMRegressor(BaseModel):
    """LSTM regressor operating on already-windowed feature sequences."""

    def __init__(
        self,
        dataset_key: str,
        model_name: str = "LSTM_fallback",
        optimizer_name: str = "adam",
        random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_key, random_seed)
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.network: Optional[nn.Module] = None
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._best_epoch = 1
        self.best_inner_mae: Optional[float] = None

    @property
    def name(self) -> str:
        return f"{self.model_name}_{self.optimizer_name}"

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tscv: Any = None,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        if len(X_train) < 8:
            raise ValueError("LSTM tuning requires at least 8 training sequences.")

        boundary = min(max(int(len(X_train) * 0.8), 2), len(X_train) - 1)
        X_fit, X_eval = X_train[:boundary], X_train[boundary:]
        y_values = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_fit, y_eval = y_values[:boundary], y_values[boundary:]
        sequence_config = config.SEQUENCE_CONFIG
        max_epochs = int(sequence_config["max_epochs"])
        patience = int(sequence_config["early_stopping_patience"])
        n_trials = n_trials or config.OPTIMIZATION["optuna_n_trials"]
        budget = config.OPTIMIZATION["max_time_per_model_s"]
        if budget is not None:
            budget /= len(config.MODELS["TFT"].get(
                "optimizers_to_compare", ["adam"]
            ))

        def objective(trial: optuna.Trial) -> float:
            params = {
                "hidden_size": trial.suggest_categorical(
                    "hidden_size", sequence_config["hidden_sizes"]
                ),
                "n_layers": trial.suggest_int(
                    "n_layers", *sequence_config["n_layers_range"]
                ),
                "dropout": trial.suggest_float(
                    "dropout", *sequence_config["dropout_range"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *sequence_config["lr_range"], log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", sequence_config["batch_size_choices"]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", *sequence_config["l2_reg_range"], log=True
                ),
                "optimizer": self.optimizer_name,
                "seed": self.random_seed + trial.number,
            }
            _, best_epoch, inner_mae = _fit_network(
                "lstm",
                X_fit,
                y_fit,
                params,
                max_epochs,
                X_eval=X_eval,
                y_eval=y_eval,
                patience=patience,
            )
            trial.set_user_attr("best_epoch", best_epoch)
            return inner_mae

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=budget,
            show_progress_bar=False,
        )
        if not study.trials:
            raise RuntimeError(f"Optuna did not run any trial for {self.name}.")

        best_trial = study.best_trial
        self._best_epoch = int(best_trial.user_attrs.get("best_epoch", max_epochs))
        self.best_inner_mae = float(best_trial.value)
        self._best_params = dict(best_trial.params)
        self._best_params["optimizer"] = self.optimizer_name
        self._best_params["selected_epochs"] = self._best_epoch
        self._best_params["inner_train_mae"] = self.best_inner_mae
        return self.get_hyperparams()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        params = dict(self._best_params)
        params["seed"] = self.random_seed
        params["optimizer"] = self.optimizer_name
        self.network, _, _ = _fit_network(
            "lstm",
            X_train,
            np.asarray(y_train),
            params,
            max_epochs=self._best_epoch,
        )
        self._target_mean = self.network._olivesmart_y_mean
        self._target_scale = self.network._olivesmart_y_scale

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.network is None:
            raise RuntimeError(f"Model {self.name} must be fitted before predict().")
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.network(X_tensor).numpy().reshape(-1)
        return predictions * self._target_scale + self._target_mean


def make_sequences(
    features: np.ndarray,
    lookback: int,
    context: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create lookback windows, using only earlier feature rows as context."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {values.shape}.")
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    if context is not None:
        context_values = np.asarray(context, dtype=np.float32)
        if context_values.ndim != 2 or context_values.shape[1] != values.shape[1]:
            raise ValueError("Sequence context must have the same feature columns.")
        if lookback > 1:
            values = np.concatenate([context_values[-(lookback - 1):], values])
    if len(values) < lookback:
        raise ValueError(
            f"Need at least {lookback} rows to build a sequence; got {len(values)}."
        )
    return np.stack([
        values[index - lookback + 1:index + 1]
        for index in range(lookback - 1, len(values))
    ])
