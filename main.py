"""
OliveSmart — Main Entry Point
==============================
Single command to run the entire experimentation pipeline.

Usage:
    python main.py                          # Run all enabled datasets and models
    python main.py --dataset FAOSTAT        # Run only the FAOSTAT dataset
    python main.py --model Ridge            # Run only Ridge across all datasets
    python main.py --dataset FAOSTAT --model XGBoost   # Specific combination
    python main.py --phase 1                # Run only Phase 1 (EDA)
    python main.py --force-rerun            # Ignore cached results and re-run all
    python main.py --list                   # List available datasets and models
    python main.py --help                   # Show this help message

Scientific guarantees:
    - Test data is NEVER used during hyperparameter optimization
    - Splits are chronological for time-series datasets
    - Each dataset is processed independently
    - All random operations use a fixed seed for reproducibility
    - No model is silently overwritten
"""

import argparse
import hashlib
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on PYTHONPATH so all imports work correctly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import configuration FIRST (before anything else)
# ---------------------------------------------------------------------------
import config

# ---------------------------------------------------------------------------
# Logger setup (must happen early so all subsequent imports can use it)
# ---------------------------------------------------------------------------
from src.logger import setup_logger

logger = setup_logger(
    name="olivesmart",
    level=config.LOGGING["level"],
    log_dir=config.LOGS_DIR,
    console=config.LOGGING["console_output"],
    file_output=config.LOGGING["file_output"],
    prefix=config.LOGGING["log_filename_prefix"],
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for flexible experiment control.

    Returns
    -------
    argparse.Namespace
        Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        prog="OliveSmart",
        description=(
            "OliveSmart — Automated ML/DL Framework for Olive Yield Prediction\n"
            "University project OLV-5 | Supervisory Regression Framework"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(config.DATASETS.keys()),
        default=None,
        help="Run experiments only for this specific dataset.",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(config.MODELS.keys()),
        default=None,
        help="Run experiments only for this specific model.",
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help=(
            "Run only a specific phase of the pipeline:\n"
            "  1: Environment check\n"
            "  2: Data loading and EDA\n"
            "  3: Preprocessing and splitting\n"
            "  4: ML models (Baseline, Ridge, RF, XGBoost)\n"
            "  5: MLP (all optimizers)\n"
            "  6: TFT/LSTM (if applicable)\n"
            "  7: Evaluation, graphs, and reports\n"
            "  8: Full pipeline (all phases)"
        ),
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Ignore cached results and re-run all experiments from scratch.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all available datasets, models, and their status.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate the pipeline without actually running experiments.",
    )

    return parser.parse_args()


def print_banner():
    """Display the OliveSmart ASCII banner and system information."""
    banner = r"""
  ___  _ _           ____                       _
 / _ \| (_)_   _____/ ___| _ __ ___   __ _ _ __| |_
| | | | | \ \ / / _ \___ \| '_ ` _ \ / _` | '__| __|
| |_| | | |\ V /  __/___) | | | | | | (_| | |  | |_
 \___/|_|_| \_/ \___|____/|_| |_| |_|\__,_|_|   \__|

        OLV-5 — Prévision du Rendement des Oliveraies
        Framework ML/DL Automatisé et Reproductible
    """
    print(banner)


def list_configuration():
    """Print the current configuration of datasets and models."""
    print("\n" + "="*60)
    print("DATASETS CONFIGURÉS")
    print("="*60)
    for key, ds in config.DATASETS.items():
        status = "✓ ACTIVÉ" if ds["enabled"] else "✗ désactivé"
        has_target = "🎯 Supervisé" if ds["has_yield_target"] else "📊 Exploratoire"
        print(f"  [{status}] {key:<15} | {has_target} | {ds['name']}")

    print("\n" + "="*60)
    print("MODÈLES CONFIGURÉS")
    print("="*60)
    for key, m in config.MODELS.items():
        status = "✓ ACTIVÉ" if m["enabled"] else "✗ désactivé"
        opt = m.get("optimizer_type") or "—"
        print(f"  [{status}] {key:<15} | Optimizer: {opt:<10} | {m['description']}")

    print("\n" + "="*60)
    print(f"Metric de classement : {config.REPORTING['ranking_metric']}")
    print(f"Essais Optuna max    : {config.OPTIMIZATION['optuna_n_trials']}")
    print(f"Seed aléatoire       : {config.OPTIMIZATION['random_seed']}")
    print("="*60 + "\n")


def check_environment():
    """
    Verify that all required dependencies are installed and log their versions.
    This is a Phase 1 health check.

    Returns
    -------
    bool
        True if all core dependencies are available, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("VÉRIFICATION DE L'ENVIRONNEMENT")
    logger.info("=" * 60)

    all_ok = True
    core_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "optuna": "optuna",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "fpdf": "fpdf2",
        "tqdm": "tqdm",
        "colorlog": "colorlog",
        "joblib": "joblib",
        "psutil": "psutil",
    }

    optional_packages = {
        "tensorflow": "tensorflow",
        "torch": "torch (PyTorch)",
        "pytorch_forecasting": "pytorch-forecasting",
        "catboost": "catboost",
    }

    logger.info("Packages principaux :")
    for module, name in core_packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "unknown")
            logger.info(f"  [OK]  {name:<30} v{version}")
        except ImportError:
            logger.error(f"  [ERR] {name:<30} NON INSTALLE")
            all_ok = False

    logger.info("\nPackages Deep Learning (optionnels) :")
    for module, name in optional_packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "unknown")
            logger.info(f"  [OK]  {name:<30} v{version}")
        except ImportError:
            logger.warning(f"  [-]   {name:<30} non disponible (fonctionnalites DL limitees)")

    # Check GPU availability
    logger.info("\nVerification GPU :")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"  [OK]  TensorFlow GPU : {len(gpus)} GPU(s) detecte(s)")
            for gpu in gpus:
                logger.info(f"        {gpu.name}")
        else:
            logger.warning("  [-]   TensorFlow GPU : aucun GPU detecte (CPU uniquement)")
    except ImportError:
        logger.warning("  [-]   TensorFlow non installe -- verification GPU ignoree")

    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  [OK]  PyTorch GPU : {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("  [-]   PyTorch GPU : CUDA non disponible")
    except ImportError:
        logger.warning("  [-]   PyTorch non installe -- verification GPU ignoree")

    # Check data directories
    logger.info("\nVerification des repertoires de donnees :")
    for ds_key, ds_config in config.DATASETS.items():
        raw_path = ds_config["raw_path"]
        if raw_path.exists():
            files = list(raw_path.glob("*"))
            # Filter out .gitkeep
            data_files = [f for f in files if f.name != ".gitkeep"]
            if data_files:
                logger.info(
                    f"  [OK]  {ds_key:<15} : {len(data_files)} fichier(s) trouve(s)"
                )
            else:
                logger.warning(
                    f"  [-]   {ds_key:<15} : repertoire vide -- "
                    f"placer les donnees dans {raw_path}"
                )
        else:
            logger.warning(
                f"  [ERR] {ds_key:<15} : repertoire manquant -- {raw_path}"
            )

    if all_ok:
        logger.info("\n[OK] Environnement principal operationnel.")
    else:
        logger.error(
            "\n[ERR] Certains packages requis sont manquants. "
            "Executez : pip install -r requirements.txt"
        )

    return all_ok


def run_pipeline(args: argparse.Namespace):
    """
    Main pipeline entry point.
    Imports are done lazily here to allow `--list` and `--help`
    to work without all dependencies installed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Apply force-rerun setting
    if args.force_rerun:
        config.RESUME["skip_completed"] = False
        config.RESUME["force_rerun"] = True
        logger.warning("Mode force-rerun activé : toutes les expériences seront relancées.")

    # Determine which datasets to process
    if args.dataset:
        datasets_to_run = {args.dataset: config.DATASETS[args.dataset]}
        logger.info(f"Mode ciblé : dataset = {args.dataset}")
    else:
        datasets_to_run = {k: v for k, v in config.DATASETS.items() if v["enabled"]}

    # Determine which models to run
    if args.model:
        models_to_run = {args.model: config.MODELS[args.model]}
        logger.info(f"Mode ciblé : modèle = {args.model}")
    else:
        models_to_run = {k: v for k, v in config.MODELS.items() if v["enabled"]}

    logger.info(f"Datasets à traiter : {list(datasets_to_run.keys())}")
    logger.info(f"Modèles à exécuter : {list(models_to_run.keys())}")

    # Phase routing
    phase = args.phase or 8  # Default: run full pipeline

    if phase == 1:
        logger.info("Phase 1 : Verification de l'environnement uniquement.")
        check_environment()
        return

    # ----------------------------------------------------------------
    # PHASE 2 — Data Loading & Exploratory Data Analysis
    # ----------------------------------------------------------------
    if phase >= 2:
        logger.info("=" * 60)
        logger.info("PHASE 2 : Chargement et Exploration des Donnees")
        logger.info("=" * 60)

        from src.data.data_loader import DataLoader
        from src.data.data_explorer import DataExplorer

        loader  = DataLoader()
        explorer = DataExplorer()

        # Store bundles and reports for downstream phases
        bundles = {}
        eda_reports = {}
        supervised_datasets = []   # datasets where regression is feasible
        exploratory_datasets = []  # datasets for EDA only

        for ds_key in datasets_to_run:
            logger.info(f"\n>>> Dataset : {ds_key}")
            try:
                # --- Load ---
                bundle = loader.load(ds_key)
                loader.save_meta(bundle)
                bundles[ds_key] = bundle

                # --- Explore ---
                report = explorer.explore(bundle, save=True)
                eda_reports[ds_key] = report

                # --- Classify dataset ---
                if report.feasibility_verdict["is_supervised_feasible"]:
                    supervised_datasets.append(ds_key)
                else:
                    exploratory_datasets.append(ds_key)

            except FileNotFoundError as exc:
                logger.error(f"[ERR] {ds_key} : {exc}")
                logger.error("     Lancez d'abord : python downloaders/download_all.py")
            except Exception as exc:
                logger.error(f"[ERR] Erreur inattendue pour {ds_key} : {exc}")

        # --- Phase 2 final summary ---
        logger.info("\n" + "=" * 60)
        logger.info("RESUME PHASE 2")
        logger.info("=" * 60)
        logger.info(f"Datasets charges avec succes : {list(bundles.keys())}")
        logger.info(f"Regression supervisee possible : {supervised_datasets}")
        logger.info(f"Analyse exploratoire seulement : {exploratory_datasets}")
        logger.info(f"Graphiques EDA : {config.GRAPHS_DIR}")
        logger.info(f"Rapports EDA   : {config.EXPERIMENTS_DIR / 'eda'}")

        if phase == 2:
            logger.info("\nPhase 2 terminee. Lancez --phase 3 pour le preprocessement.")
            return

    # ----------------------------------------------------------------
    # PHASE 3+ (stubs — will be implemented in future phases)
    # ----------------------------------------------------------------
    if phase >= 3:
        logger.info("=" * 60)
        logger.info("PHASE 3 : Preprocessement et Partitions")
        logger.info("=" * 60)

        from src.data.data_loader import DataLoader
        from src.data.preprocessing import Preprocessor
        from src.data.data_splitter import DataSplitter

        loader   = DataLoader()
        splitter = DataSplitter()

        # We only preprocess datasets that are supervised-feasible
        # For now that is FAOSTAT only (determined in Phase 2)
        supervised_keys = [
            k for k, v in config.DATASETS.items()
            if v["enabled"] and v.get("has_yield_target", False)
            and k in datasets_to_run
        ]
        logger.info(f"Datasets supervises a preprocesser : {supervised_keys}")

        prepared_data = {}   # key -> dict with X/y splits + pipeline info

        for ds_key in supervised_keys:
            logger.info(f"\n>>> Preprocessement : {ds_key}")
            try:
                bundle = loader.load(ds_key)
                prep   = Preprocessor(dataset_key=ds_key)

                # Dataset-specific preparation
                if ds_key == "FAOSTAT":
                    df_prep = prep.prepare_faostat(bundle.df, add_lags=True, n_lags=1)
                    target_col = "Yield_hg_ha"
                else:
                    ds_cfg     = config.DATASETS[ds_key]
                    target_col = bundle.meta.get("detected_target") or ds_cfg["target_column"]
                    df_prep    = prep.prepare_generic(bundle.df, target_col)

                # Leakage audit before split
                suspicious = prep.audit_leakage(df_prep, target_col)
                if suspicious:
                    logger.warning(f"  Risque de fuite : {suspicious}")

                # Split
                split = splitter.split(
                    df=df_prep,
                    target_col=target_col,
                    date_col=config.DATASETS[ds_key].get("date_column"),
                    strategy="chronological",
                    dataset_key=ds_key,
                )
                splitter.save(split)

                # Build and apply pipelines for each model
                model_splits = {}
                for model_name, model_cfg in models_to_run.items():
                    if model_name == "Baseline":
                        # Baseline needs no preprocessing
                        model_splits[model_name] = {
                            "X_train": split.X_train.values,
                            "X_val":   split.X_val.values,
                            "X_test":  split.X_test.values,
                            "y_train": split.y_train,
                            "y_val":   split.y_val,
                            "y_test":  split.y_test,
                        }
                        continue

                    pipe = prep.build_pipeline(model_name, split.X_train)
                    X_tr, X_v, X_te = prep.fit_transform_split(
                        pipe, split.X_train, split.X_val, split.X_test,
                        model_type=model_name
                    )
                    prep.save_pipeline_info(pipe, model_name, split.X_train)
                    model_splits[model_name] = {
                        "X_train": X_tr, "X_val": X_v, "X_test": X_te,
                        "y_train": split.y_train,
                        "y_val":   split.y_val,
                        "y_test":  split.y_test,
                        "pipeline": pipe,
                    }

                prepared_data[ds_key] = {
                    "split":        split,
                    "model_splits": model_splits,
                    "preprocessor": prep,
                }

                logger.info(f"[OK] {ds_key} : preprocessement termine.")
                logger.info(f"     Train={split.split_meta['n_train']} | "
                            f"Val={split.split_meta['n_val']} | "
                            f"Test={split.split_meta['n_test']}")

            except Exception as exc:
                logger.error(f"[ERR] Preprocessement {ds_key} : {exc}")

        if phase == 3:
            logger.info("\nPhase 3 terminee. Lancez --phase 4 pour les modeles ML.")
            return

    if phase >= 4:
        logger.info("=" * 60)
        logger.info("PHASE 4 : Entrainement des modeles ML classiques")
        logger.info("=" * 60)

        from src.models.classical_models import (
            CLASSICAL_MODELS,
            build_classical_model,
        )
        import json
        import pandas as pd

        classical_models = {
            name: cfg for name, cfg in models_to_run.items()
            if name in CLASSICAL_MODELS
        }
        deferred_models = sorted(set(models_to_run) - CLASSICAL_MODELS)
        if deferred_models:
            logger.warning(
                f"Modeles hors phase 4 ignores : {deferred_models}. "
                "MLP et TFT/LSTM relevent des phases suivantes."
            )

        phase4_results = []
        results_dir = config.EXPERIMENTS_DIR / "model_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for ds_key, prepared in prepared_data.items():
            split = prepared["split"]
            preprocessor = prepared["preprocessor"]

            for model_name in classical_models:
                logger.info(f"\n>>> Entrainement {ds_key} / {model_name}")
                try:
                    result_path = results_dir / f"{ds_key}_{model_name}.json"
                    model_path = (
                        config.MODELS_DIR / ds_key / f"{model_name}.joblib"
                    )
                    prediction_path = (
                        config.EXPERIMENTS_DIR / "predictions"
                        / f"{ds_key}_{model_name}_predictions.csv"
                    )

                    fingerprint = hashlib.sha256()
                    for features, target in (
                        (split.X_train, split.y_train),
                        (split.X_val, split.y_val),
                        (split.X_test, split.y_test),
                    ):
                        fingerprint.update(
                            json.dumps(list(features.columns)).encode("utf-8")
                        )
                        fingerprint.update(
                            pd.util.hash_pandas_object(
                                features, index=True
                            ).values.tobytes()
                        )
                        fingerprint.update(
                            pd.util.hash_pandas_object(
                                target, index=True
                            ).values.tobytes()
                        )
                    fingerprint.update(json.dumps({
                        "model": config.MODELS[model_name],
                        "optimization": config.OPTIMIZATION,
                    }, sort_keys=True, default=str).encode("utf-8"))
                    run_fingerprint = fingerprint.hexdigest()

                    can_resume = (
                        config.RESUME["skip_completed"]
                        and not config.RESUME["force_rerun"]
                        and result_path.is_file()
                        and model_path.is_file()
                        and prediction_path.is_file()
                    )
                    if can_resume:
                        try:
                            with open(result_path, "r", encoding="utf-8") as result_file:
                                cached_result = json.load(result_file)
                        except (OSError, json.JSONDecodeError) as exc:
                            logger.warning(
                                f"Resultat en cache illisible ({result_path}) : {exc}"
                            )
                        else:
                            if (
                                cached_result.get("status") == "completed"
                                and cached_result.get("run_fingerprint")
                                == run_fingerprint
                            ):
                                logger.info(
                                    f"Resultat deja termine, reutilise : {result_path}"
                                )
                                phase4_results.append(cached_result)
                                continue

                    preprocessing_pipeline = None
                    if model_name != "Baseline":
                        preprocessing_pipeline = preprocessor.build_pipeline(
                            model_name, split.X_train
                        )

                    model = build_classical_model(
                        model_name=model_name,
                        dataset_key=ds_key,
                        preprocessing_pipeline=preprocessing_pipeline,
                        random_seed=config.OPTIMIZATION["random_seed"],
                    )

                    if config.MODELS[model_name].get("optimizer_type") == "classical":
                        model.optimize(
                            split.X_train,
                            split.y_train,
                            split.tscv,
                        )

                    model.fit_timed(split.X_train, split.y_train)
                    result = model.evaluate_all_splits(
                        split.X_train, split.y_train,
                        split.X_val, split.y_val,
                        split.X_test, split.y_test,
                    )
                    result["status"] = "completed"
                    result["best_params"] = model.get_hyperparams()
                    result["cv_mae"] = model.best_cv_mae
                    result["run_fingerprint"] = run_fingerprint

                    if model_path.exists():
                        logger.warning(
                            f"Le modele existant sera remplace : {model_path}"
                        )
                    saved_model_path = model.save()
                    result["model_path"] = str(
                        saved_model_path.relative_to(config.ROOT_DIR)
                    )

                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    predictions = pd.concat([
                        pd.DataFrame({
                            "split": "validation",
                            "actual": split.y_val.to_numpy(),
                            "predicted": model.predict(split.X_val),
                        }),
                        pd.DataFrame({
                            "split": "test",
                            "actual": split.y_test.to_numpy(),
                            "predicted": model.predict(split.X_test),
                        }),
                    ], ignore_index=True)
                    predictions.to_csv(prediction_path, index=False)
                    result["result_path"] = str(
                        result_path.relative_to(config.ROOT_DIR)
                    )
                    result["prediction_path"] = str(
                        prediction_path.relative_to(config.ROOT_DIR)
                    )
                    with open(result_path, "w", encoding="utf-8") as result_file:
                        json.dump(result, result_file, indent=2, ensure_ascii=False)
                    phase4_results.append(result)
                    logger.info(f"[OK] Resultats sauvegardes : {result_path}")
                except Exception:
                    logger.exception(
                        f"[ERR] Echec phase 4 pour {ds_key} / {model_name}"
                    )

        if phase4_results:
            summary_path = config.EXPERIMENTS_DIR / "phase4_model_comparison.csv"
            summary = pd.DataFrame(phase4_results)
            ranking_metric = config.REPORTING["ranking_metric"]
            if ranking_metric in summary.columns:
                summary = summary.sort_values(
                    ranking_metric,
                    ascending=config.REPORTING["ranking_direction"] == "minimize",
                )
            summary.to_csv(summary_path, index=False)
            logger.info(f"\nComparaison phase 4 : {summary_path}")
            logger.info(
                f"Modeles termines : {len(phase4_results)}"
            )
        elif not classical_models:
            logger.warning("Aucun modele classique selectionne pour la phase 4.")
        else:
            logger.warning(
                "Aucun modele phase 4 n'a ete entraine. "
                "Consultez les erreurs ci-dessus et verifiez les donnees."
            )

        if phase == 4:
            logger.info("\nPhase 4 terminee.")
            return

    if phase >= 5:
        logger.warning(
            "Les phases 5 a 7 (MLP, TFT/LSTM, evaluation finale et rapports) "
            "ne sont pas encore implementees."
        )

    logger.info("\nPipeline OliveSmart termine.")


def main():
    """
    Application entry point.
    Parses arguments, prints the banner, and routes execution.
    """
    args = parse_arguments()
    print_banner()

    if args.list:
        list_configuration()
        return

    logger.info("Démarrage d'OliveSmart...")
    logger.info(f"Répertoire racine : {config.ROOT_DIR}")

    # Always run environment check at startup
    env_ok = check_environment()

    if not env_ok:
        logger.error(
            "Arret : l'environnement n'est pas correctement configure. "
            "Installez les dependances manquantes et relancez."
        )
        sys.exit(1)

    if not args.dry_run:
        run_pipeline(args)
    else:
        logger.info("Mode dry-run : simulation uniquement, aucune experience lancee.")


if __name__ == "__main__":
    main()
