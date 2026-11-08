"""Persistence helpers shared by the deep-learning experiment phases."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import config
from src.data.data_splitter import SplitResult
from src.models.base_model import BaseModel
from src.logger import get_logger

logger = get_logger("olivesmart")


def experiment_fingerprint(
    split: SplitResult,
    model_name: str,
    parameters: Dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    for features, target in (
        (split.X_train, split.y_train),
        (split.X_val, split.y_val),
        (split.X_test, split.y_test),
    ):
        digest.update(json.dumps(list(features.columns)).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(features, index=True).values.tobytes()
        )
        digest.update(
            pd.util.hash_pandas_object(target, index=True).values.tobytes()
        )
    digest.update(json.dumps({
        "model": model_name,
        "parameters": parameters,
    }, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def load_completed_result(
    result_path: Path,
    model_path: Path,
    prediction_path: Path,
    fingerprint: str,
) -> Optional[Dict[str, Any]]:
    if not (
        config.RESUME["skip_completed"]
        and not config.RESUME["force_rerun"]
        and result_path.is_file()
        and model_path.is_file()
        and prediction_path.is_file()
    ):
        return None

    try:
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Resultat en cache illisible ({result_path}) : {exc}")
        return None
    if (
        result.get("status") == "completed"
        and result.get("run_fingerprint") == fingerprint
    ):
        return result
    return None


def write_status_result(
    result_path: Path,
    dataset_key: str,
    model_name: str,
    status: str,
    reason: str,
    fingerprint: str,
    requested_model: Optional[str] = None,
) -> Dict[str, Any]:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": dataset_key,
        "model": model_name,
        "requested_model": requested_model or model_name,
        "status": status,
        "reason": reason,
        "run_fingerprint": fingerprint,
    }
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def save_model_result(
    model: BaseModel,
    metrics: Dict[str, Any],
    X_val: Any,
    y_val: Any,
    X_test: Any,
    y_test: Any,
    result_path: Path,
    prediction_path: Path,
    fingerprint: str,
    requested_model: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_path = config.MODELS_DIR / model.dataset_key / f"{model.name}.joblib"
    if model_path.exists():
        logger.warning(f"Le modèle existant sera remplacé : {model_path}")
    model_path = model.save()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat([
        pd.DataFrame({
            "split": "validation",
            "actual": np.asarray(y_val).reshape(-1),
            "predicted": model.predict(X_val),
        }),
        pd.DataFrame({
            "split": "test",
            "actual": np.asarray(y_test).reshape(-1),
            "predicted": model.predict(X_test),
        }),
    ], ignore_index=True)
    predictions.to_csv(prediction_path, index=False)

    result = dict(metrics)
    result.update({
        "status": "completed",
        "requested_model": requested_model or model.name,
        "run_fingerprint": fingerprint,
        "model_path": str(model_path.relative_to(config.ROOT_DIR)),
        "result_path": str(result_path.relative_to(config.ROOT_DIR)),
        "prediction_path": str(prediction_path.relative_to(config.ROOT_DIR)),
    })
    if extra:
        result.update(extra)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result
