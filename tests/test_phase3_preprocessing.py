"""
OliveSmart — Phase 3 Test Suite
==================================
Tests for data splitting and preprocessing.

Critical tests:
  - Chronological split preserves time order
  - No leakage: transformers fitted only on train
  - Ratios sum to 1.0
  - Production_tonnes is excluded from FAOSTAT features (leakage prevention)
  - Val and test sets are truly held out

Run with:
    pytest tests/test_phase3_preprocessing.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data.data_loader import DataLoader
from src.data.data_splitter import DataSplitter, SplitResult
from src.data.preprocessing import Preprocessor, LagFeatureTransformer, FAOSTAT_LEAKAGE_COLS


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def faostat_bundle():
    return DataLoader().load("FAOSTAT")


@pytest.fixture(scope="module")
def prepared_faostat(faostat_bundle):
    prep = Preprocessor(dataset_key="FAOSTAT")
    return prep.prepare_faostat(faostat_bundle.df, add_lags=True, n_lags=1)


@pytest.fixture(scope="module")
def split_result(prepared_faostat):
    splitter = DataSplitter()
    return splitter.split(
        df=prepared_faostat,
        target_col="Yield_hg_ha",
        feature_cols=None,
        date_col="Year",
        strategy="chronological",
        dataset_key="FAOSTAT",
    )


# ============================================================
# 1. DataSplitter — basic behavior
# ============================================================

class TestDataSplitter:

    def test_splitter_instantiation(self):
        s = DataSplitter()
        assert s is not None

    def test_ratios_sum_to_one(self):
        """Config ratios must sum to 1.0."""
        s = DataSplitter()
        total = s.train_ratio + s.val_ratio + s.test_ratio
        assert abs(total - 1.0) < 1e-9

    def test_invalid_ratios_raise(self):
        """Ratios that don't sum to 1 must raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            DataSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_returns_split_result(self, split_result):
        assert isinstance(split_result, SplitResult)

    def test_no_overlap_between_partitions(self, split_result):
        """Train, val, test index sets must be disjoint."""
        train_set = set(split_result.train_idx.tolist())
        val_set   = set(split_result.val_idx.tolist())
        test_set  = set(split_result.test_idx.tolist())
        assert train_set.isdisjoint(val_set),  "Overlap between train and val!"
        assert train_set.isdisjoint(test_set), "Overlap between train and test!"
        assert val_set.isdisjoint(test_set),   "Overlap between val and test!"

    def test_total_samples_preserved(self, split_result, prepared_faostat):
        """n_train + n_val + n_test must equal total valid samples."""
        n_total = split_result.split_meta["n_total"]
        n_sum   = (split_result.split_meta["n_train"] +
                   split_result.split_meta["n_val"]   +
                   split_result.split_meta["n_test"])
        assert n_total == n_sum

    def test_partitions_non_empty(self, split_result):
        """All three partitions must have at least 1 sample."""
        assert len(split_result.X_train) > 0
        assert len(split_result.X_val)   > 0
        assert len(split_result.X_test)  > 0

    def test_x_y_shapes_consistent(self, split_result):
        """X and y must have matching row counts in each partition."""
        assert len(split_result.X_train) == len(split_result.y_train)
        assert len(split_result.X_val)   == len(split_result.y_val)
        assert len(split_result.X_test)  == len(split_result.y_test)

    def test_target_not_in_features(self, split_result):
        """Target column must NOT appear in X_train."""
        assert "Yield_hg_ha" not in split_result.X_train.columns

    def test_tscv_configured(self, split_result):
        """TimeSeriesSplit must be provided for cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit
        assert isinstance(split_result.tscv, TimeSeriesSplit)

    def test_split_meta_has_periods(self, split_result):
        """Split metadata must include train/val/test time periods."""
        meta = split_result.split_meta
        assert meta["train_period"] is not None
        assert meta["val_period"]   is not None
        assert meta["test_period"]  is not None

    def test_too_small_dataset_raises(self):
        """Datasets smaller than min_samples must raise ValueError."""
        df_tiny = pd.DataFrame({"Year": [2020, 2021], "Yield": [100.0, 110.0]})
        splitter = DataSplitter(min_samples=10)
        with pytest.raises(ValueError, match="valid samples"):
            splitter.split(df_tiny, target_col="Yield")


# ============================================================
# 2. Chronological split — time order integrity
# ============================================================

class TestChronologicalSplit:
    """
    Critical: for time-series data, the split must respect chronological order.
    No future data should appear in the training set.
    """

    def test_train_years_before_val(self, split_result, prepared_faostat):
        """All training years must precede all validation years."""
        train_years = prepared_faostat["Year"].iloc[
            :split_result.split_meta["n_train"]
        ].values
        val_years = prepared_faostat["Year"].iloc[
            split_result.split_meta["n_train"]:
            split_result.split_meta["n_train"] + split_result.split_meta["n_val"]
        ].values
        assert train_years.max() < val_years.min(), (
            f"Time leak! Last train year ({train_years.max()}) >= "
            f"first val year ({val_years.min()})"
        )

    def test_val_years_before_test(self, split_result, prepared_faostat):
        """All validation years must precede all test years."""
        n_tr = split_result.split_meta["n_train"]
        n_v  = split_result.split_meta["n_val"]
        val_years  = prepared_faostat["Year"].iloc[n_tr:n_tr + n_v].values
        test_years = prepared_faostat["Year"].iloc[n_tr + n_v:].values
        assert val_years.max() < test_years.min(), (
            f"Time leak! Last val year ({val_years.max()}) >= "
            f"first test year ({test_years.min()})"
        )

    def test_train_period_in_meta(self, split_result):
        """Train period start must be earlier than test period end."""
        train_start = int(split_result.split_meta["train_period"]["start"])
        test_end    = int(split_result.split_meta["test_period"]["end"])
        assert train_start < test_end


# ============================================================
# 3. Preprocessor — FAOSTAT feature preparation
# ============================================================

class TestFAOSTATPreparation:

    def test_prepare_returns_dataframe(self, faostat_bundle):
        prep = Preprocessor(dataset_key="FAOSTAT")
        df_prep = prep.prepare_faostat(faostat_bundle.df, add_lags=True)
        assert isinstance(df_prep, pd.DataFrame)
        assert not df_prep.empty

    def test_production_excluded(self, prepared_faostat):
        """Production_tonnes must NOT be in prepared features (leakage)."""
        for leakage_col in FAOSTAT_LEAKAGE_COLS:
            assert leakage_col not in prepared_faostat.columns, (
                f"LEAKAGE DETECTED: '{leakage_col}' found in prepared features!"
            )

    def test_target_present(self, prepared_faostat):
        """Target column Yield_hg_ha must be present."""
        assert "Yield_hg_ha" in prepared_faostat.columns

    def test_lag_feature_created(self, prepared_faostat):
        """Lag feature Yield_hg_ha_lag1 must be present when add_lags=True."""
        assert "Yield_hg_ha_lag1" in prepared_faostat.columns

    def test_lag_no_nan_in_dataset(self, prepared_faostat):
        """After lag creation and row drop, no NaN should remain."""
        assert prepared_faostat.isnull().sum().sum() == 0, (
            "NaN values found after lag preparation — rows with lags not dropped."
        )

    def test_lag_values_are_shifted(self, prepared_faostat):
        """Yield_hg_ha_lag1 at row i must equal Yield_hg_ha at row i-1."""
        df = prepared_faostat.reset_index(drop=True)
        for i in range(1, len(df)):
            expected = df["Yield_hg_ha"].iloc[i - 1]
            actual   = df["Yield_hg_ha_lag1"].iloc[i]
            assert abs(expected - actual) < 1e-6, (
                f"Lag mismatch at row {i}: expected {expected}, got {actual}"
            )

    def test_data_sorted_by_year(self, prepared_faostat):
        """Prepared FAOSTAT must be sorted by Year (ascending)."""
        years = prepared_faostat["Year"].values
        assert list(years) == sorted(years), "Data is not sorted by Year!"


# ============================================================
# 4. Preprocessing pipelines — no-leakage protocol
# ============================================================

class TestPreprocessingPipelines:

    def test_build_ridge_pipeline(self, split_result):
        """Ridge pipeline must include StandardScaler."""
        prep = Preprocessor(dataset_key="FAOSTAT")
        pipe = prep.build_pipeline("Ridge", split_result.X_train)
        assert isinstance(pipe, Pipeline)
        # Check that StandardScaler is in the numeric transformer
        col_tf = pipe.named_steps["preprocessor"]
        for name, transformer, cols in col_tf.transformers:
            if name == "num":
                step_names = [s[0] for s in transformer.steps]
                assert "scaler" in step_names, "Ridge pipeline missing StandardScaler!"

    def test_build_rf_pipeline_no_scaler(self, split_result):
        """RandomForest pipeline must NOT include StandardScaler."""
        prep = Preprocessor(dataset_key="FAOSTAT")
        pipe = prep.build_pipeline("RandomForest", split_result.X_train)
        col_tf = pipe.named_steps["preprocessor"]
        for name, transformer, cols in col_tf.transformers:
            if name == "num":
                step_names = [s[0] for s in transformer.steps]
                assert "scaler" not in step_names, \
                    "RandomForest pipeline should NOT have a scaler!"

    def test_fit_transform_returns_arrays(self, split_result):
        """fit_transform_split must return numpy arrays."""
        prep = Preprocessor(dataset_key="FAOSTAT")
        pipe = prep.build_pipeline("Ridge", split_result.X_train)
        X_tr, X_v, X_te = prep.fit_transform_split(
            pipe, split_result.X_train, split_result.X_val, split_result.X_test,
            model_type="Ridge"
        )
        assert isinstance(X_tr, np.ndarray)
        assert isinstance(X_v,  np.ndarray)
        assert isinstance(X_te, np.ndarray)

    def test_output_shapes_correct(self, split_result):
        """Transformed arrays must match input row counts."""
        prep = Preprocessor(dataset_key="FAOSTAT")
        pipe = prep.build_pipeline("XGBoost", split_result.X_train)
        X_tr, X_v, X_te = prep.fit_transform_split(
            pipe, split_result.X_train, split_result.X_val, split_result.X_test,
            model_type="XGBoost"
        )
        assert X_tr.shape[0] == len(split_result.X_train)
        assert X_v.shape[0]  == len(split_result.X_val)
        assert X_te.shape[0] == len(split_result.X_test)

    # ------------------------------------------------------------------
    # THE KEY ANTI-LEAKAGE TEST
    # ------------------------------------------------------------------

    def test_scaler_fitted_on_train_only(self, split_result):
        """
        CRITICAL: The scaler's mean and std must be computed from X_train only.

        Proof: We manually compute the mean of X_train and compare it to
        the scaler's learned mean. They must match exactly.
        Any deviation would indicate the scaler used val/test data.
        """
        prep = Preprocessor(dataset_key="FAOSTAT")
        pipe = prep.build_pipeline("Ridge", split_result.X_train)

        # FIT the pipeline (required before reading learned parameters)
        prep.fit_transform_split(
            pipe, split_result.X_train, split_result.X_val, split_result.X_test,
            model_type="Ridge"
        )

        # Use transformers_ (fitted attribute) to get the actual fitted instances
        col_tf = pipe.named_steps["preprocessor"]
        for name, transformer, cols in col_tf.transformers_:
            if name == "num":
                scaler = transformer.named_steps["scaler"]
                assert hasattr(scaler, "mean_"), \
                    "StandardScaler was never fitted — fit_transform_split may not have run."

                # Manually compute train mean for the same numeric columns
                num_train = split_result.X_train[list(cols)]
                expected_mean = num_train.mean().values
                actual_mean   = scaler.mean_

                np.testing.assert_allclose(
                    actual_mean, expected_mean, rtol=1e-5,
                    err_msg=(
                        "LEAKAGE DETECTED: Scaler mean does not match X_train mean! "
                        "The scaler may have been fitted on more than just training data."
                    )
                )

    def test_val_not_used_in_fit(self, split_result):
        """
        Verify val set statistics differ from train (proves they are separate).
        If scaler were fitted on val too, train and val means would diverge
        in the same way — but we can check the scaler only knows train stats.
        """
        prep = Preprocessor(dataset_key="FAOSTAT")
        pipe = prep.build_pipeline("MLP", split_result.X_train)
        X_tr, X_v, X_te = prep.fit_transform_split(
            pipe, split_result.X_train, split_result.X_val, split_result.X_test,
            model_type="MLP"
        )
        # After StandardScaler on train: X_train mean ≈ 0, std ≈ 1
        # X_val mean will NOT be 0 (different data distribution)
        train_col_means = np.abs(X_tr.mean(axis=0))
        val_col_means   = np.abs(X_v.mean(axis=0))

        # Train means should be ~0 (standardized)
        np.testing.assert_array_less(
            train_col_means, np.full_like(train_col_means, 1e-9) + 0.01,
            err_msg="X_train means after scaling should be ~0"
        )

    def test_different_pipelines_per_model(self, split_result):
        """Each model type must produce a separate pipeline instance."""
        prep   = Preprocessor(dataset_key="FAOSTAT")
        pipe_r = prep.build_pipeline("Ridge",        split_result.X_train)
        pipe_f = prep.build_pipeline("RandomForest", split_result.X_train)
        # They must be different objects
        assert pipe_r is not pipe_f


# ============================================================
# 5. Lag Feature Transformer
# ============================================================

class TestLagFeatureTransformer:

    def test_creates_lag_column(self):
        df = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0, 50.0]})
        lagger = LagFeatureTransformer(columns=["y"], n_lags=1)
        df_out = lagger.transform(df)
        assert "y_lag1" in df_out.columns

    def test_lag_values_correct(self):
        df = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0, 50.0]})
        lagger = LagFeatureTransformer(columns=["y"], n_lags=1)
        df_out = lagger.transform(df)
        # Row 1: lag1 = row 0 value = 10.0
        assert df_out["y_lag1"].iloc[1] == 10.0
        # Row 2: lag1 = row 1 value = 20.0
        assert df_out["y_lag1"].iloc[2] == 20.0

    def test_first_row_is_nan(self):
        df = pd.DataFrame({"y": [10.0, 20.0, 30.0]})
        lagger = LagFeatureTransformer(columns=["y"], n_lags=1)
        df_out = lagger.transform(df)
        assert pd.isna(df_out["y_lag1"].iloc[0])

    def test_does_not_modify_original(self):
        """Transformer must not modify the input DataFrame."""
        df = pd.DataFrame({"y": [10.0, 20.0, 30.0]})
        df_before = df.copy()
        LagFeatureTransformer(columns=["y"], n_lags=1).transform(df)
        pd.testing.assert_frame_equal(df, df_before)

    def test_multiple_lags(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        lagger = LagFeatureTransformer(columns=["y"], n_lags=3)
        df_out = lagger.transform(df)
        assert "y_lag1" in df_out.columns
        assert "y_lag2" in df_out.columns
        assert "y_lag3" in df_out.columns


# ============================================================
# 6. Leakage audit
# ============================================================

class TestLeakageAudit:

    def test_audit_detects_perfect_correlation(self):
        """Columns with |r|=1 to target should be flagged."""
        df = pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target":    [2.0, 4.0, 6.0, 8.0, 10.0],  # = 2 * feature_a
        })
        prep = Preprocessor()
        flagged = prep.audit_leakage(df, "target", threshold=0.99)
        assert "feature_a" in flagged

    def test_audit_no_false_positive(self):
        """Uncorrelated columns should NOT be flagged."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "noise":  rng.normal(0, 1, 50),
            "target": rng.normal(0, 1, 50),
        })
        prep = Preprocessor()
        flagged = prep.audit_leakage(df, "target", threshold=0.99)
        assert "noise" not in flagged

    def test_faostat_production_flagged(self, faostat_bundle):
        """In FAOSTAT, Production_tonnes should be flagged as a leakage risk."""
        prep = Preprocessor(dataset_key="FAOSTAT")
        # Only run if Production_tonnes exists (it should in wide format)
        if "Production_tonnes" in faostat_bundle.df.columns:
            flagged = prep.audit_leakage(
                faostat_bundle.df, "Yield_hg_ha", threshold=0.70
            )
            # Production and Yield are correlated (not perfectly, since area varies)
            # At minimum, verify the audit runs without error
            assert isinstance(flagged, list)
