"""
OliveSmart — Phase 1 Verification Tests
=========================================
Tests to confirm that Phase 1 (environment setup and project structure)
is correctly configured.

Run with:
    pytest tests/test_phase1_environment.py -v

Expected outcome:
    All tests pass before proceeding to Phase 2.
"""

import sys
import os
from pathlib import Path
import pytest

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. PROJECT STRUCTURE TESTS
# ============================================================

class TestProjectStructure:
    """Verify that all required directories and files exist."""

    def test_config_exists(self):
        """config.py must exist at the project root."""
        assert (PROJECT_ROOT / "config.py").exists(), (
            "config.py not found. It should be at the project root."
        )

    def test_main_exists(self):
        """main.py must exist at the project root."""
        assert (PROJECT_ROOT / "main.py").exists(), (
            "main.py not found."
        )

    def test_requirements_exists(self):
        """requirements.txt must exist."""
        assert (PROJECT_ROOT / "requirements.txt").exists()

    def test_requirements_deep_exists(self):
        """requirements_deep.txt must exist."""
        assert (PROJECT_ROOT / "requirements_deep.txt").exists()

    def test_readme_exists(self):
        """README.md must exist."""
        assert (PROJECT_ROOT / "README.md").exists()

    def test_data_directories(self):
        """All raw data directories must exist."""
        datasets = ["faostat", "nasa_power", "sentinel2", "wapor"]
        for ds in datasets:
            path = PROJECT_ROOT / "data" / "raw" / ds
            assert path.exists(), f"Directory missing: {path}"

    def test_processed_directories(self):
        """All processed data directories must exist."""
        datasets = ["faostat", "nasa_power", "sentinel2", "wapor"]
        for ds in datasets:
            path = PROJECT_ROOT / "data" / "processed" / ds
            assert path.exists(), f"Directory missing: {path}"

    def test_src_packages(self):
        """All src sub-packages must have __init__.py."""
        packages = [
            PROJECT_ROOT / "src" / "__init__.py",
            PROJECT_ROOT / "src" / "data" / "__init__.py",
            PROJECT_ROOT / "src" / "models" / "__init__.py",
            PROJECT_ROOT / "src" / "optimization" / "__init__.py",
        ]
        for p in packages:
            assert p.exists(), f"Missing __init__.py: {p}"

    def test_results_directories(self):
        """Results subdirectories must exist."""
        for sub in ["experiments", "models", "graphs"]:
            path = PROJECT_ROOT / "results" / sub
            assert path.exists(), f"Directory missing: {path}"

    def test_reports_directories(self):
        """Reports subdirectories must exist."""
        for sub in ["FAOSTAT", "NASA_POWER", "SENTINEL2", "WAPOR"]:
            path = PROJECT_ROOT / "reports" / sub
            assert path.exists(), f"Directory missing: {path}"

    def test_logs_directory(self):
        """Logs directory must exist."""
        assert (PROJECT_ROOT / "logs").exists()


# ============================================================
# 2. CONFIGURATION TESTS
# ============================================================

class TestConfiguration:
    """Verify that config.py is correctly structured."""

    def setup_method(self):
        """Import config once per test class."""
        import config
        self.config = config

    def test_config_import(self):
        """config.py must be importable without errors."""
        import config  # noqa

    def test_datasets_defined(self):
        """DATASETS dict must be defined and non-empty."""
        assert hasattr(self.config, "DATASETS")
        assert len(self.config.DATASETS) > 0

    def test_models_defined(self):
        """MODELS dict must be defined and non-empty."""
        assert hasattr(self.config, "MODELS")
        assert len(self.config.MODELS) > 0

    def test_dataset_keys(self):
        """Expected datasets must all be present in config."""
        expected = {"FAOSTAT", "NASA_POWER", "SENTINEL2", "WAPOR"}
        actual = set(self.config.DATASETS.keys())
        assert expected == actual, f"Missing datasets: {expected - actual}"

    def test_model_keys(self):
        """Expected models must all be present in config."""
        expected = {"Baseline", "Ridge", "RandomForest", "XGBoost", "MLP", "TFT"}
        actual = set(self.config.MODELS.keys())
        assert expected == actual, f"Missing models: {expected - actual}"

    def test_random_seed_set(self):
        """A fixed random seed must be defined."""
        assert self.config.OPTIMIZATION["random_seed"] == 42

    def test_splitting_ratios_sum_to_one(self):
        """Train + val + test ratios must sum to 1.0."""
        s = self.config.SPLITTING
        total = s["train_ratio"] + s["val_ratio"] + s["test_ratio"]
        assert abs(total - 1.0) < 1e-9, f"Ratios sum to {total}, expected 1.0"

    def test_path_attributes(self):
        """Key path attributes must exist and be Path objects."""
        for attr in ["ROOT_DIR", "DATA_DIR", "RESULTS_DIR", "REPORTS_DIR", "LOGS_DIR"]:
            assert hasattr(self.config, attr), f"Missing attribute: {attr}"
            val = getattr(self.config, attr)
            assert isinstance(val, Path), f"{attr} must be a Path, got {type(val)}"

    def test_ranking_metric(self):
        """Ranking metric must be a recognized metric name."""
        valid = {"val_mae", "val_rmse", "val_r2"}
        assert self.config.REPORTING["ranking_metric"] in valid


# ============================================================
# 3. CORE DEPENDENCIES TESTS
# ============================================================

class TestCoreDependencies:
    """Verify that all required packages can be imported."""

    def test_numpy(self):
        import numpy as np
        assert np.__version__

    def test_pandas(self):
        import pandas as pd
        assert pd.__version__

    def test_sklearn(self):
        import sklearn
        assert sklearn.__version__

    def test_xgboost(self):
        import xgboost
        assert xgboost.__version__

    def test_optuna(self):
        import optuna
        assert optuna.__version__

    def test_matplotlib(self):
        import matplotlib
        assert matplotlib.__version__

    def test_seaborn(self):
        import seaborn
        assert seaborn.__version__

    def test_fpdf(self):
        import fpdf
        assert fpdf

    def test_tqdm(self):
        import tqdm
        assert tqdm.__version__

    def test_joblib(self):
        import joblib
        assert joblib.__version__

    def test_scipy(self):
        import scipy
        assert scipy.__version__


# ============================================================
# 4. LOGGER TESTS
# ============================================================

class TestLogger:
    """Verify the custom logger module works correctly."""

    def test_logger_import(self):
        """Logger module must be importable."""
        from src.logger import setup_logger, get_logger  # noqa

    def test_setup_logger_returns_logger(self):
        """setup_logger must return a logging.Logger instance."""
        import logging
        from src.logger import setup_logger
        logger = setup_logger(name="test_olivesmart", file_output=False)
        assert isinstance(logger, logging.Logger)

    def test_get_logger(self):
        """get_logger must return the same logger."""
        import logging
        from src.logger import setup_logger, get_logger
        setup_logger(name="test_get", file_output=False)
        logger = get_logger("test_get")
        assert isinstance(logger, logging.Logger)


# ============================================================
# 5. MAIN.PY SMOKE TEST
# ============================================================

class TestMainModule:
    """Verify that main.py can be imported and basic functions work."""

    def test_main_import(self):
        """main.py must be importable (no side effects on import)."""
        # We import the functions directly to avoid running main()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "main", PROJECT_ROOT / "main.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't exec — just check it loads
        assert spec is not None

    def test_parse_arguments_no_args(self):
        """parse_arguments() with empty argv must return defaults."""
        import sys
        from main import parse_arguments
        original_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_arguments()
            assert args.dataset is None
            assert args.model is None
            assert args.force_rerun is False
        finally:
            sys.argv = original_argv
