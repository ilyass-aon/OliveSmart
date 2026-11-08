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
        logger.info("Phase 1 : Vérification de l'environnement uniquement.")
        check_environment()
        return

    if phase >= 2:
        logger.info("Phase 2 : Chargement et exploration des données.")
        logger.warning(
            "⚠ Phase 2 non encore implémentée. "
            "Exécutez `python main.py --phase 1` pour vérifier l'environnement."
        )
        # Phase 2 will be implemented in the next session:
        # from src.data.data_loader import DataLoader
        # from src.data.data_explorer import DataExplorer
        # ...

    logger.info("Pipeline OliveSmart terminé.")


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
