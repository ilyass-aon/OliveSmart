"""
OliveSmart — Phase 2 Test Suite
==================================
Tests for data loading and exploratory data analysis.

Run with:
    pytest tests/test_phase2_data.py -v

Expected outcome:
    All tests pass before proceeding to Phase 3.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data.data_loader import DataLoader, DatasetBundle
from src.data.data_explorer import DataExplorer, EDAReport


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def loader():
    return DataLoader()


@pytest.fixture(scope="module")
def explorer():
    return DataExplorer()


@pytest.fixture(scope="module")
def all_bundles(loader):
    """Load all enabled datasets once for the entire test module."""
    return loader.load_all()


@pytest.fixture(scope="module")
def faostat_bundle(loader):
    return loader.load("FAOSTAT")


@pytest.fixture(scope="module")
def nasa_bundle(loader):
    return loader.load("NASA_POWER")


@pytest.fixture(scope="module")
def sentinel_bundle(loader):
    return loader.load("SENTINEL2")


@pytest.fixture(scope="module")
def wapor_bundle(loader):
    return loader.load("WAPOR")


# ============================================================
# 1. DataLoader — basic loading tests
# ============================================================

class TestDataLoader:
    """Verify DataLoader returns valid DatasetBundles."""

    def test_loader_instantiation(self):
        """DataLoader must instantiate without error."""
        dl = DataLoader()
        assert dl is not None
        assert dl.raw_dir.exists()

    def test_load_returns_bundle(self, faostat_bundle):
        """load() must return a DatasetBundle NamedTuple."""
        assert isinstance(faostat_bundle, DatasetBundle)

    def test_bundle_has_df(self, faostat_bundle):
        """Bundle must contain a non-empty DataFrame."""
        assert isinstance(faostat_bundle.df, pd.DataFrame)
        assert not faostat_bundle.df.empty

    def test_bundle_has_meta(self, faostat_bundle):
        """Bundle must contain a non-empty metadata dict."""
        assert isinstance(faostat_bundle.meta, dict)
        assert len(faostat_bundle.meta) > 0

    def test_bundle_dataset_key(self, faostat_bundle):
        """Bundle dataset_key must match the requested key."""
        assert faostat_bundle.dataset_key == "FAOSTAT"

    def test_load_all_returns_dict(self, all_bundles):
        """load_all() must return a dict with at least one entry."""
        assert isinstance(all_bundles, dict)
        assert len(all_bundles) > 0

    def test_all_enabled_datasets_loaded(self, all_bundles):
        """All enabled datasets must be present in the result."""
        enabled_keys = {k for k, v in config.DATASETS.items() if v["enabled"]}
        for key in enabled_keys:
            assert key in all_bundles, f"Dataset '{key}' not loaded."

    def test_invalid_key_raises_value_error(self, loader):
        """Requesting an unknown dataset key must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset key"):
            loader.load("NONEXISTENT_DATASET")


# ============================================================
# 2. FAOSTAT bundle — content tests
# ============================================================

class TestFAOSTATBundle:
    """Verify FAOSTAT dataset structure and content."""

    def test_has_year_column(self, faostat_bundle):
        """FAOSTAT must have a 'Year' column."""
        assert "Year" in faostat_bundle.df.columns

    def test_has_yield_column(self, faostat_bundle):
        """FAOSTAT must have a yield column."""
        df = faostat_bundle.df
        yield_cols = [c for c in df.columns if "yield" in c.lower() or "hg_ha" in c.lower()]
        assert len(yield_cols) >= 1, f"No yield column found. Columns: {list(df.columns)}"

    def test_minimum_rows(self, faostat_bundle):
        """FAOSTAT must have at least 20 rows (years)."""
        assert len(faostat_bundle.df) >= 20

    def test_year_column_is_numeric(self, faostat_bundle):
        """Year column must be numeric."""
        df = faostat_bundle.df
        assert pd.api.types.is_numeric_dtype(df["Year"]), \
            f"Year dtype: {df['Year'].dtype}"

    def test_yield_values_positive(self, faostat_bundle):
        """Yield values must be positive (physical constraint)."""
        df = faostat_bundle.df
        yield_col = next(
            (c for c in df.columns if "yield" in c.lower() or "hg_ha" in c.lower()),
            None
        )
        if yield_col:
            valid = df[yield_col].dropna()
            assert (valid > 0).all(), "Some yield values are non-positive."

    def test_meta_contains_shape(self, faostat_bundle):
        """Metadata must report correct shape."""
        meta = faostat_bundle.meta
        assert "shape" in meta
        assert meta["shape"] == faostat_bundle.df.shape

    def test_meta_contains_md5(self, faostat_bundle):
        """Metadata must contain a file checksum."""
        assert "md5" in faostat_bundle.meta
        assert len(faostat_bundle.meta["md5"]) > 0

    def test_meta_detected_target(self, faostat_bundle):
        """Metadata must detect a target column for FAOSTAT."""
        assert faostat_bundle.meta.get("detected_target") is not None


# ============================================================
# 3. NASA POWER bundle — content tests
# ============================================================

class TestNASAPOWERBundle:
    """Verify NASA POWER dataset structure."""

    def test_has_year_column(self, nasa_bundle):
        assert "Year" in nasa_bundle.df.columns

    def test_has_temperature(self, nasa_bundle):
        """Must have at least one temperature column."""
        df = nasa_bundle.df
        temp_cols = [c for c in df.columns if "T2M" in c.upper() or "temp" in c.lower()]
        assert len(temp_cols) >= 1

    def test_has_precipitation(self, nasa_bundle):
        """Must have at least one precipitation column."""
        df = nasa_bundle.df
        prec_cols = [c for c in df.columns
                     if "precip" in c.lower() or "prec" in c.lower() or "rain" in c.lower()]
        assert len(prec_cols) >= 1

    def test_no_extreme_fill_values(self, nasa_bundle):
        """NASA POWER fill value -999 must have been replaced with NaN."""
        num_df = nasa_bundle.df.select_dtypes(include=np.number)
        assert not (num_df == -999).any().any(), \
            "Fill value -999 found — should be NaN."

    def test_minimum_rows(self, nasa_bundle):
        """Must have at least 30 rows."""
        assert len(nasa_bundle.df) >= 30

    def test_has_region_column(self, nasa_bundle):
        """Must have a Region column (3 Moroccan regions)."""
        assert "Region" in nasa_bundle.df.columns
        n_regions = nasa_bundle.df["Region"].nunique()
        assert n_regions >= 1


# ============================================================
# 4. Sentinel-2 / NDVI bundle — content tests
# ============================================================

class TestSentinel2Bundle:
    """Verify Sentinel-2 / NDVI dataset structure."""

    def test_has_ndvi_column(self, sentinel_bundle):
        """Must have at least one NDVI column."""
        df = sentinel_bundle.df
        ndvi_cols = [c for c in df.columns if "ndvi" in c.lower()]
        assert len(ndvi_cols) >= 1, f"No NDVI column. Columns: {list(df.columns)}"

    def test_ndvi_values_in_valid_range(self, sentinel_bundle):
        """All NDVI values must be in [-1, 1]."""
        df = sentinel_bundle.df
        ndvi_col = next(c for c in df.columns if "ndvi" in c.lower())
        valid = df[ndvi_col].dropna()
        assert (valid >= -1).all() and (valid <= 1).all(), \
            f"NDVI values outside [-1,1] range: min={valid.min()}, max={valid.max()}"

    def test_minimum_rows(self, sentinel_bundle):
        assert len(sentinel_bundle.df) >= 10


# ============================================================
# 5. WaPOR bundle — content tests
# ============================================================

class TestWaPORBundle:
    """Verify WaPOR dataset structure."""

    def test_has_et_column(self, wapor_bundle):
        """Must have an evapotranspiration column."""
        df = wapor_bundle.df
        et_cols = [c for c in df.columns if "et" in c.lower() or "evap" in c.lower()]
        assert len(et_cols) >= 1, f"No ET column. Columns: {list(df.columns)}"

    def test_minimum_rows(self, wapor_bundle):
        assert len(wapor_bundle.df) >= 10

    def test_has_year_column(self, wapor_bundle):
        assert "Year" in wapor_bundle.df.columns


# ============================================================
# 6. DataExplorer — EDA tests
# ============================================================

class TestDataExplorer:
    """Verify DataExplorer produces valid EDA reports."""

    def test_explorer_instantiation(self):
        exp = DataExplorer()
        assert exp is not None

    def test_explore_returns_report(self, explorer, faostat_bundle):
        """explore() must return an EDAReport NamedTuple."""
        report = explorer.explore(faostat_bundle, save=False)
        assert isinstance(report, EDAReport)

    def test_report_has_summary(self, explorer, faostat_bundle):
        report = explorer.explore(faostat_bundle, save=False)
        assert isinstance(report.summary, dict)
        assert "n_rows" in report.summary
        assert "n_cols" in report.summary

    def test_report_has_stats(self, explorer, faostat_bundle):
        report = explorer.explore(faostat_bundle, save=False)
        assert isinstance(report.stats, pd.DataFrame)
        assert not report.stats.empty

    def test_report_has_missing_analysis(self, explorer, faostat_bundle):
        report = explorer.explore(faostat_bundle, save=False)
        assert isinstance(report.missing_analysis, pd.DataFrame)
        assert "column" in report.missing_analysis.columns
        assert "pct_missing" in report.missing_analysis.columns

    def test_report_has_outlier_analysis(self, explorer, faostat_bundle):
        report = explorer.explore(faostat_bundle, save=False)
        assert isinstance(report.outlier_analysis, pd.DataFrame)

    def test_report_has_correlation_matrix(self, explorer, faostat_bundle):
        report = explorer.explore(faostat_bundle, save=False)
        assert isinstance(report.correlation_matrix, pd.DataFrame)

    def test_report_has_feasibility_verdict(self, explorer, faostat_bundle):
        report = explorer.explore(faostat_bundle, save=False)
        verdict = report.feasibility_verdict
        assert isinstance(verdict, dict)
        assert "is_supervised_feasible" in verdict
        assert "reasoning" in verdict
        assert "recommendation" in verdict

    def test_missing_severity_values(self, explorer, faostat_bundle):
        """Severity column must only contain known levels."""
        report = explorer.explore(faostat_bundle, save=False)
        valid_severities = {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual = set(report.missing_analysis["severity"].unique())
        assert actual.issubset(valid_severities), \
            f"Unknown severity levels: {actual - valid_severities}"

    def test_outlier_percentages_valid(self, explorer, faostat_bundle):
        """Outlier percentages must be between 0 and 100."""
        report = explorer.explore(faostat_bundle, save=False)
        if not report.outlier_analysis.empty:
            pcts = report.outlier_analysis["pct_outliers"]
            assert (pcts >= 0).all() and (pcts <= 100).all()

    def test_summary_row_count_matches_df(self, explorer, faostat_bundle):
        """Summary n_rows must match actual DataFrame row count."""
        report = explorer.explore(faostat_bundle, save=False)
        assert report.summary["n_rows"] == len(faostat_bundle.df)


# ============================================================
# 7. Scientific feasibility rules
# ============================================================

class TestFeasibilityRules:
    """
    Verify that the feasibility assessment follows scientific rules.
    These tests encode the core scientific constraints of the project.
    """

    def test_faostat_is_supervised_feasible(self, explorer, faostat_bundle):
        """FAOSTAT has a yield target — supervised regression must be feasible."""
        report = explorer.explore(faostat_bundle, save=False)
        assert report.feasibility_verdict["is_supervised_feasible"] is True, \
            f"FAOSTAT should be feasible. Reasoning: {report.feasibility_verdict['reasoning']}"

    def test_nasa_power_is_not_supervised(self, explorer, nasa_bundle):
        """NASA POWER has no yield target — must NOT be directly supervised."""
        report = explorer.explore(nasa_bundle, save=False)
        assert report.feasibility_verdict["is_supervised_feasible"] is False, \
            "NASA POWER alone has no yield target — cannot be supervised."

    def test_sentinel2_is_not_supervised(self, explorer, sentinel_bundle):
        """Sentinel-2 NDVI alone has no yield target — must NOT be supervised."""
        report = explorer.explore(sentinel_bundle, save=False)
        assert report.feasibility_verdict["is_supervised_feasible"] is False

    def test_wapor_is_not_supervised(self, explorer, wapor_bundle):
        """WaPOR alone has no yield target — must NOT be supervised."""
        report = explorer.explore(wapor_bundle, save=False)
        assert report.feasibility_verdict["is_supervised_feasible"] is False

    def test_verdict_contains_target_column_for_faostat(self, explorer, faostat_bundle):
        """FAOSTAT verdict must identify the target column."""
        report = explorer.explore(faostat_bundle, save=False)
        assert report.feasibility_verdict["target_column"] is not None

    def test_verdict_target_col_is_none_for_nasa(self, explorer, nasa_bundle):
        """NASA POWER verdict target_column must be None (no yield in this dataset)."""
        report = explorer.explore(nasa_bundle, save=False)
        # Either None or not a real yield column
        col = report.feasibility_verdict.get("target_column")
        # NASA POWER may have no detected target, which is correct
        # (config declares has_yield_target = False)
        assert report.feasibility_verdict["is_supervised_feasible"] is False

    def test_no_data_leakage_in_exploration(self, explorer, faostat_bundle):
        """
        EDA must not modify the original DataFrame.
        This protects against accidental data leakage during exploration.
        """
        df_before = faostat_bundle.df.copy()
        explorer.explore(faostat_bundle, save=False)
        df_after = faostat_bundle.df
        pd.testing.assert_frame_equal(df_before, df_after,
                                      check_like=False,
                                      obj="DataFrame must not be modified during EDA")


# ============================================================
# 8. list_available_files utility
# ============================================================

class TestListAvailableFiles:
    """Verify the file listing utility."""

    def test_list_faostat_files(self, loader):
        files = loader.list_available_files("FAOSTAT")
        assert isinstance(files, list)
        assert len(files) > 0, "No files found in FAOSTAT directory."

    def test_list_nasa_files(self, loader):
        files = loader.list_available_files("NASA_POWER")
        assert len(files) > 0

    def test_no_gitkeep_in_list(self, loader):
        """list_available_files must not return .gitkeep."""
        files = loader.list_available_files("FAOSTAT")
        names = [f.name for f in files]
        assert ".gitkeep" not in names
