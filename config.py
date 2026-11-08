"""
OliveSmart — Central Configuration File
========================================
This file is the single source of truth for all experiment parameters.
Modify values here to control the behavior of the entire framework
without touching any other source file.

Scientific constraints enforced:
- No test data leakage during hyperparameter search
- Chronological splits for time-series datasets
- Independent processing per dataset
- Fixed random seed for reproducibility
"""

import os
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# Root directory of the project (where this file lives)
ROOT_DIR = Path(__file__).parent.resolve()

# Data directories
DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

# Output directories
RESULTS_DIR     = ROOT_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
MODELS_DIR      = RESULTS_DIR / "models"
GRAPHS_DIR      = RESULTS_DIR / "graphs"

# Reports
REPORTS_DIR     = ROOT_DIR / "reports"

# Logs
LOGS_DIR        = ROOT_DIR / "logs"

# ============================================================
# DATASETS CONFIGURATION
# ============================================================
# Each dataset has:
#   - name: human-readable identifier used in reports
#   - raw_path: expected location of raw data file(s)
#   - has_yield_target: whether supervised ML is scientifically valid
#   - target_column: name of the yield column (if any)
#   - date_column: name of the time column (if any)
#   - enabled: whether to include this dataset in the pipeline
#   - notes: scientific notes for the report

DATASETS = {
    "FAOSTAT": {
        "name": "FAOSTAT — Données Historiques Maroc",
        "raw_path": RAW_DATA_DIR / "faostat",
        "has_yield_target": True,
        "target_column": "Yield_hg_ha",       # Wide format column name
        "target_unit": "hg/ha",             # FAOSTAT uses hectograms per hectare
        "date_column": "Year",
        "enabled": True,
        "notes": (
            "Données nationales agrégées pour le Maroc. "
            "Environ 60 observations (1961–2023). "
            "Modèles temporels limités par la taille réduite du dataset."
        ),
    },
    "NASA_POWER": {
        "name": "NASA POWER — Données Climatiques",
        "raw_path": RAW_DATA_DIR / "nasa_power",
        "has_yield_target": False,          # No yield column in raw NASA data
        "target_column": None,
        "date_column": "YEAR",
        "enabled": True,
        "notes": (
            "Données climatiques sans variable de rendement intégrée. "
            "Analyse exploratoire uniquement sauf fusion avec FAOSTAT."
        ),
    },
    "SENTINEL2": {
        "name": "Sentinel-2 — Indices de Végétation (NDVI)",
        "raw_path": RAW_DATA_DIR / "sentinel2",
        "has_yield_target": False,
        "target_column": None,
        "date_column": "date",
        "enabled": True,
        "notes": (
            "Données satellitaires en mode CSV manuel (sans Google Earth Engine). "
            "Indices de végétation (NDVI, EVI, etc.) par région/période. "
            "Analyse exploratoire et corrélation avec rendement si fusion disponible."
        ),
    },
    "WAPOR": {
        "name": "FAO WaPOR — Données Hydriques",
        "raw_path": RAW_DATA_DIR / "wapor",
        "has_yield_target": False,
        "target_column": None,
        "date_column": "date",
        "enabled": True,
        "notes": (
            "Données d'évapotranspiration et de bilan hydrique. "
            "Analyse exploratoire uniquement sauf fusion géotemporelle."
        ),
    },
}

# ============================================================
# MODELS CONFIGURATION
# ============================================================
# enabled: whether to run this model
# requires_scaling: whether to standardize features before training
# optimizer_type: "classical" (GridSearch/RandomSearch) or "optuna"

MODELS = {
    "Baseline": {
        "enabled": True,
        "requires_scaling": False,
        "optimizer_type": None,             # No hyperparameter search needed
        "description": "Mean baseline — reference model",
    },
    "Ridge": {
        "enabled": True,
        "requires_scaling": True,           # Ridge is sensitive to feature scale
        "optimizer_type": "classical",
        "description": "Ridge Regression with L2 regularization",
    },
    "RandomForest": {
        "enabled": True,
        "requires_scaling": False,          # Tree-based: scale-invariant
        "optimizer_type": "classical",
        "description": "Random Forest Regressor",
    },
    "XGBoost": {
        "enabled": True,
        "requires_scaling": False,          # Tree-based: scale-invariant
        "optimizer_type": "classical",
        "description": "XGBoost Gradient Boosting Regressor",
    },
    "MLP": {
        "enabled": True,
        "requires_scaling": True,           # Neural nets need standardization
        "optimizer_type": "optuna",
        "optimizers_to_compare": ["adam", "adamw", "sgd"],
        "min_observations_required": 50,
        "description": "Multilayer Perceptron (PyTorch — Adam/AdamW/SGD compare)",
    },
    "TFT": {
        "enabled": False,                   # Disabled by default (< 1 week deadline)
        "requires_scaling": True,
        "optimizer_type": "optuna",
        "optimizers_to_compare": ["adam", "adamw"],
        "min_observations_required": 100,   # TFT needs sufficient temporal data
        "fallback_to_lstm": True,           # Use LSTM if data insufficient for TFT
        "description": "Temporal Fusion Transformer (PyTorch Forecasting)",
    },
}

# ============================================================
# HYPERPARAMETER SEARCH CONFIGURATION
# ============================================================

OPTIMIZATION = {
    # Maximum number of Optuna trials per model (MLP, LSTM fallback)
    "optuna_n_trials": 30,

    # Maximum wall-clock time per model (seconds); None = no limit
    "max_time_per_model_s": 600,            # 10 minutes max per model

    # Cross-validation strategy for classical models
    # Options: "timeseries" (TimeSeriesSplit) | "kfold" | "none"
    "cv_strategy": "timeseries",
    "cv_n_splits": 5,

    # Random state for reproducibility
    "random_seed": 42,

    # GridSearchCV / RandomizedSearchCV
    "randomized_n_iter": 50,                # Iterations for RandomizedSearchCV
    "n_jobs": -1,                           # Use all CPU cores for sklearn search
}

# ============================================================
# DATA SPLITTING CONFIGURATION
# ============================================================

SPLITTING = {
    # For time-series datasets (chronological order preserved)
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # Minimum samples required for a valid train/val/test split
    "min_samples_for_split": 20,          # FAOSTAT has 32 effective rows after lag

    # Random seed for non-temporal splits
    "random_seed": 42,
}

# ============================================================
# MLP (DEEP LEARNING) CONFIGURATION
# ============================================================

MLP_CONFIG = {
    # Architecture search space for Optuna
    "n_layers_range": (2, 5),
    "units_choices": [32, 64, 128, 256, 512],
    "dropout_range": (0.0, 0.5),
    "lr_range": (1e-5, 1e-2),
    "batch_size_choices": [16, 32, 64, 128],
    "l2_reg_range": (1e-6, 1e-2),

    # Training control
    "max_epochs": 300,
    "early_stopping_patience": 20,
    "early_stopping_monitor": "val_loss",
    "restore_best_weights": True,
}

# ============================================================
# SEQUENCE MODEL (LSTM FALLBACK) CONFIGURATION
# ============================================================

SEQUENCE_CONFIG = {
    "lookback": 3,
    "minimum_training_sequences": 12,
    "hidden_sizes": [16, 32, 64],
    "n_layers_range": (1, 2),
    "dropout_range": (0.0, 0.4),
    "lr_range": (1e-4, 1e-2),
    "batch_size_choices": [8, 16, 32],
    "l2_reg_range": (1e-6, 1e-2),
    "max_epochs": 200,
    "early_stopping_patience": 15,
}

# ============================================================
# REPORTING CONFIGURATION
# ============================================================

REPORTING = {
    # Language for PDF reports ("fr" or "en")
    "report_language": "fr",

    # Code comments language (for reference only, not runtime)
    "code_comment_language": "en",

    # Include training metrics in reports (not just val/test)
    "include_train_metrics": True,

    # Generate per-model graphs
    "generate_graphs": True,

    # Save format for graphs
    "graph_dpi": 150,
    "graph_format": "png",

    # Metric used to rank models and select the best
    "ranking_metric": "val_mae",            # Lower is better
    "ranking_direction": "minimize",
}

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOGGING = {
    # Log level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
    "level": "INFO",

    # Whether to also print to console (always True in CLI mode)
    "console_output": True,

    # Whether to save logs to file
    "file_output": True,

    # Log filename pattern (timestamp appended automatically)
    "log_filename_prefix": "olivesmart",
}

# ============================================================
# EXPERIMENT RESUME CONFIGURATION
# ============================================================

RESUME = {
    # If True, skip experiments whose JSON result file already exists
    # and has status "completed". Set to False to force re-run everything.
    "skip_completed": True,

    # If True, overwrite existing result files
    "force_rerun": False,
}

# ============================================================
# SYSTEM CONFIGURATION
# ============================================================

SYSTEM = {
    # GPU usage for TensorFlow/PyTorch (None = auto-detect)
    # Set to -1 to force CPU only
    "gpu_id": None,

    # Memory growth for TensorFlow GPU (prevents OOM)
    "tf_memory_growth": True,

    # Library versions snapshot (auto-populated at runtime)
    "log_library_versions": True,
}
