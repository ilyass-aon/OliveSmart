"""
OliveSmart — Data Loader Module
=================================
Responsible for loading raw datasets from disk into pandas DataFrames.

Design principles:
- Each dataset has a dedicated loader method with dataset-specific logic
- A generic fallback loader handles arbitrary CSV/Excel files
- All loaders return a DatasetBundle (NamedTuple) with the DataFrame + metadata
- Encoding detection prevents UnicodeDecodeError on Windows
- No data transformation happens here — only loading and minimal type coercion

Usage:
    from src.data.data_loader import DataLoader

    loader = DataLoader()
    bundle = loader.load("FAOSTAT")
    print(bundle.df.head())
    print(bundle.meta)
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, NamedTuple, Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.logger import get_logger

logger = get_logger("olivesmart")


# ============================================================
# DatasetBundle — the return type for all loaders
# ============================================================

class DatasetBundle(NamedTuple):
    """
    Container returned by every load_* method.

    Attributes
    ----------
    dataset_key : str
        Identifier matching config.DATASETS keys (e.g. "FAOSTAT").
    df : pd.DataFrame
        The loaded raw DataFrame (no transformations applied).
    meta : dict
        Metadata about the loaded data:
            - file_path: path of the primary file loaded
            - file_size_kb: file size in kilobytes
            - shape: (n_rows, n_cols)
            - columns: list of column names
            - dtypes: dict of column -> dtype
            - encoding: detected file encoding
            - loaded_at: ISO timestamp
            - md5: file checksum for reproducibility
            - n_files: number of files loaded (if multiple)
    """
    dataset_key: str
    df: pd.DataFrame
    meta: Dict[str, Any]


# ============================================================
# DataLoader — main class
# ============================================================

class DataLoader:
    """
    Loads raw datasets from disk.

    Each supported dataset has a dedicated method (load_faostat,
    load_nasa_power, etc.) that knows the expected file names and
    structure.  A generic load() dispatcher routes by dataset key.

    Parameters
    ----------
    raw_dir : Path, optional
        Root directory containing raw dataset subdirectories.
        Defaults to config.RAW_DATA_DIR.
    """

    # Candidate encodings to try in order (UTF-8-sig handles BOM files
    # exported by Excel on Windows)
    _ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]

    # Candidate target column names (heuristic yield detection)
    _YIELD_KEYWORDS = [
        "yield", "rendement", "kg_ha", "hg_ha", "production_par_ha",
        "yield_hg_ha", "value"
    ]

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or config.RAW_DATA_DIR
        logger.debug(f"DataLoader initialized. Raw dir: {self.raw_dir}")

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def load(self, dataset_key: str) -> DatasetBundle:
        """
        Load a dataset by its configuration key.

        Parameters
        ----------
        dataset_key : str
            One of "FAOSTAT", "NASA_POWER", "SENTINEL2", "WAPOR".

        Returns
        -------
        DatasetBundle

        Raises
        ------
        ValueError
            If the dataset key is not recognized.
        FileNotFoundError
            If no data files are found in the expected directory.
        """
        loaders = {
            "FAOSTAT":    self.load_faostat,
            "NASA_POWER": self.load_nasa_power,
            "SENTINEL2":  self.load_sentinel2,
            "WAPOR":      self.load_wapor,
        }
        if dataset_key not in loaders:
            raise ValueError(
                f"Unknown dataset key: '{dataset_key}'. "
                f"Valid keys: {list(loaders.keys())}"
            )
        logger.info(f"Loading dataset: {dataset_key}")
        bundle = loaders[dataset_key]()
        logger.info(
            f"[OK] {dataset_key} loaded — "
            f"{bundle.df.shape[0]} rows x {bundle.df.shape[1]} cols"
        )
        return bundle

    def load_all(self) -> Dict[str, DatasetBundle]:
        """
        Load all enabled datasets defined in config.DATASETS.

        Returns
        -------
        dict
            Mapping from dataset_key -> DatasetBundle.
            Datasets that fail to load are logged but not re-raised.
        """
        bundles = {}
        for key, ds_cfg in config.DATASETS.items():
            if not ds_cfg["enabled"]:
                logger.info(f"Skipping disabled dataset: {key}")
                continue
            try:
                bundles[key] = self.load(key)
            except Exception as exc:
                logger.error(f"Failed to load {key}: {exc}")
        return bundles

    # ------------------------------------------------------------------
    # Dataset-specific loaders
    # ------------------------------------------------------------------

    def load_faostat(self) -> DatasetBundle:
        """
        Load FAOSTAT Morocco Olives data.

        Preferred file: faostat_morocco_olives_wide.csv (one row per year).
        Fallback: faostat_morocco_olives_long.csv (pivots automatically).
        Also attempts to find any CSV in the faostat/ folder matching
        common FAOSTAT export naming patterns.

        Returns
        -------
        DatasetBundle
        """
        ds_dir = self.raw_dir / "faostat"
        priority_files = [
            "faostat_morocco_olives_wide.csv",
            "faostat_morocco_olives_long.csv",
        ]

        df, file_path = self._load_priority_csv(
            ds_dir, priority_files,
            glob_patterns=["*.csv", "*.xlsx", "*.xls"]
        )

        # If the loaded file is in long format, pivot to wide
        if "Element" in df.columns and "Value" in df.columns:
            logger.debug("FAOSTAT: detected long format — pivoting to wide.")
            df = self._pivot_faostat_long(df)

        # Standardize column names
        df = self._standardize_faostat_columns(df)

        # Ensure Year is integer
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

        meta = self._build_meta("FAOSTAT", df, file_path)
        return DatasetBundle("FAOSTAT", df, meta)

    def load_nasa_power(self) -> DatasetBundle:
        """
        Load NASA POWER climate data.

        Preferred: nasa_power_annual_morocco.csv (annual aggregates).
        Fallback: nasa_power_monthly_morocco.csv (monthly, more detailed).
        """
        ds_dir = self.raw_dir / "nasa_power"
        priority_files = [
            "nasa_power_annual_morocco.csv",
            "nasa_power_monthly_morocco.csv",
        ]

        df, file_path = self._load_priority_csv(
            ds_dir, priority_files,
            glob_patterns=["*.csv"]
        )

        # Coerce numeric columns
        num_cols = [c for c in df.columns if c not in
                    ("Region", "YearMonth", "Date", "Source")]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace common fill values
        df = df.replace(-999.0, np.nan)
        df = df.replace(-999, np.nan)

        meta = self._build_meta("NASA_POWER", df, file_path)
        return DatasetBundle("NASA_POWER", df, meta)

    def load_sentinel2(self) -> DatasetBundle:
        """
        Load Sentinel-2 / NDVI vegetation index data.

        Preferred: ndvi_annual_morocco.csv (annual NDVI stats).
        Fallback: ndvi_monthly_morocco_olive_regions.csv (monthly).
        Also looks for any CSV containing 'ndvi' in its name.
        """
        ds_dir = self.raw_dir / "sentinel2"
        priority_files = [
            "ndvi_annual_morocco.csv",
            "modis_ndvi_annual_morocco.csv",
            "ndvi_monthly_morocco_olive_regions.csv",
            "modis_ndvi_morocco_olive_regions.csv",
        ]

        df, file_path = self._load_priority_csv(
            ds_dir, priority_files,
            glob_patterns=["*ndvi*.csv", "*sentinel*.csv", "*.csv"]
        )

        # Coerce NDVI values
        ndvi_cols = [c for c in df.columns if "ndvi" in c.lower()]
        for col in ndvi_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Clip to valid NDVI range [-1, 1]
            df[col] = df[col].clip(-1.0, 1.0)

        meta = self._build_meta("SENTINEL2", df, file_path)
        return DatasetBundle("SENTINEL2", df, meta)

    def load_wapor(self) -> DatasetBundle:
        """
        Load FAO WaPOR evapotranspiration / water productivity data.

        Preferred: wapor_et_annual_morocco.csv.
        """
        ds_dir = self.raw_dir / "wapor"
        priority_files = [
            "wapor_et_annual_morocco.csv",
            "wapor_aeti_morocco.csv",
        ]

        df, file_path = self._load_priority_csv(
            ds_dir, priority_files,
            glob_patterns=["*.csv", "*.xlsx"]
        )

        # Coerce numeric columns
        num_cols = [c for c in df.columns if c not in
                    ("Region", "Date", "Source", "RasterCode", "DataType",
                     "Unit", "Level")]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        meta = self._build_meta("WAPOR", df, file_path)
        return DatasetBundle("WAPOR", df, meta)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_priority_csv(
        self,
        directory: Path,
        priority_files: List[str],
        glob_patterns: Optional[List[str]] = None,
    ):
        """
        Try to load files in priority order.
        Falls back to glob patterns if none of the priority files exist.

        Returns
        -------
        (pd.DataFrame, Path)
        """
        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}\n"
                f"Run: python downloaders/download_all.py"
            )

        # 1. Try priority files first
        for fname in priority_files:
            fpath = directory / fname
            if fpath.exists():
                logger.debug(f"Loading priority file: {fpath.name}")
                df = self._read_file(fpath)
                return df, fpath

        # 2. Try glob patterns
        if glob_patterns:
            for pattern in glob_patterns:
                matches = sorted(directory.glob(pattern))
                # Filter .gitkeep
                matches = [m for m in matches if m.name != ".gitkeep"]
                if matches:
                    fpath = matches[0]
                    logger.warning(
                        f"Priority files not found in {directory.name}/. "
                        f"Using first match: {fpath.name}"
                    )
                    df = self._read_file(fpath)
                    return df, fpath

        raise FileNotFoundError(
            f"No data files found in {directory}.\n"
            f"Expected files: {priority_files}\n"
            f"Run: python downloaders/download_all.py"
        )

    def _read_file(self, path: Path) -> pd.DataFrame:
        """
        Read a CSV or Excel file, trying multiple encodings.

        Parameters
        ----------
        path : Path
            Path to the file to read.

        Returns
        -------
        pd.DataFrame
        """
        suffix = path.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            logger.debug(f"Reading Excel: {path.name}")
            return pd.read_excel(path)

        # CSV: try encodings in order
        for enc in self._ENCODINGS:
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                logger.debug(f"Read CSV '{path.name}' with encoding '{enc}'")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as exc:
                raise RuntimeError(f"Failed to read {path}: {exc}") from exc

        raise UnicodeDecodeError(
            f"Could not decode {path} with any of {self._ENCODINGS}"
        )

    def _pivot_faostat_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot FAOSTAT long-format DataFrame to wide format.
        Long format: (Area, Item, Element, Year, Unit, Value)
        Wide format: (Year, Yield_hg_ha, Production_tonnes, Area_harvested_ha)
        """
        try:
            df_wide = df.pivot_table(
                index=["Area", "Item", "Year"] if "Area" in df.columns else ["Year"],
                columns="Element",
                values="Value",
                aggfunc="first",
            ).reset_index()
            df_wide.columns.name = None
            return df_wide
        except Exception as exc:
            logger.warning(f"FAOSTAT pivot failed ({exc}). Returning long format.")
            return df

    def _standardize_faostat_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize FAOSTAT column names for consistent downstream processing.
        Maps various possible column names to canonical names.
        """
        rename_map = {
            # From wide format (download_all.py output)
            "Yield_hg_ha":          "Yield_hg_ha",
            "Area_harvested_ha":    "Area_harvested_ha",
            "Production_tonnes":    "Production_tonnes",
            # From pivoted long format
            "Yield":                "Yield_hg_ha",
            "Area harvested":       "Area_harvested_ha",
            "Production":           "Production_tonnes",
            # FAOSTAT official export naming
            "Area Harvested":       "Area_harvested_ha",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df

    def _build_meta(
        self, dataset_key: str, df: pd.DataFrame, file_path: Path
    ) -> Dict[str, Any]:
        """
        Build a metadata dictionary for a loaded DataFrame.

        Parameters
        ----------
        dataset_key : str
        df : pd.DataFrame
        file_path : Path

        Returns
        -------
        dict
        """
        # Compute file checksum for reproducibility
        try:
            md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
            file_size_kb = file_path.stat().st_size / 1024
        except Exception:
            md5 = "unknown"
            file_size_kb = 0.0

        # Detect potential target column using heuristics
        target_col = self._detect_target_column(df, dataset_key)

        return {
            "dataset_key":    dataset_key,
            "file_path":      str(file_path),
            "file_size_kb":   round(file_size_kb, 2),
            "shape":          df.shape,
            "n_rows":         df.shape[0],
            "n_cols":         df.shape[1],
            "columns":        list(df.columns),
            "dtypes":         {c: str(df[c].dtype) for c in df.columns},
            "encoding":       "utf-8-sig",
            "loaded_at":      datetime.now().isoformat(),
            "md5":            md5,
            "detected_target": target_col,
            "config_target":  config.DATASETS[dataset_key].get("target_column"),
        }

    def _detect_target_column(
        self, df: pd.DataFrame, dataset_key: str
    ) -> Optional[str]:
        """
        Heuristically detect the yield target column.

        Strategy:
        1. Check config.DATASETS for a specified target_column.
        2. Search column names for yield-related keywords.
        3. Return None if no match.

        Parameters
        ----------
        df : pd.DataFrame
        dataset_key : str

        Returns
        -------
        str or None
        """
        # 1. Config override
        cfg_target = config.DATASETS.get(dataset_key, {}).get("target_column")
        if cfg_target and cfg_target in df.columns:
            return cfg_target

        # 2. Keyword search (case-insensitive)
        lower_cols = {c.lower(): c for c in df.columns}
        for kw in self._YIELD_KEYWORDS:
            if kw in lower_cols:
                return lower_cols[kw]

        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_available_files(self, dataset_key: str) -> List[Path]:
        """
        List all data files found in the raw directory for a dataset.

        Parameters
        ----------
        dataset_key : str

        Returns
        -------
        list of Path
        """
        ds_cfg = config.DATASETS.get(dataset_key, {})
        raw_path = ds_cfg.get("raw_path", self.raw_dir / dataset_key.lower())
        if not raw_path.exists():
            return []
        files = [
            f for f in raw_path.iterdir()
            if f.is_file() and f.name != ".gitkeep"
        ]
        return sorted(files)

    def save_meta(self, bundle: DatasetBundle, out_dir: Optional[Path] = None):
        """
        Save the metadata of a DatasetBundle to a JSON file.

        Parameters
        ----------
        bundle : DatasetBundle
        out_dir : Path, optional
            Target directory. Defaults to results/experiments/.
        """
        out_dir = out_dir or config.EXPERIMENTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{bundle.dataset_key}_load_meta.json"

        # Convert non-serializable values
        meta_serializable = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in bundle.meta.items()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta_serializable, f, indent=2, ensure_ascii=False)
        logger.debug(f"Metadata saved: {out_path}")
