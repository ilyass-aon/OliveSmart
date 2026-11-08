"""Phase 7 model comparison, plots, and PDF reports."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

import config
from src.logger import get_logger

logger = get_logger("olivesmart")


def _read_model_results(
    results_dir: Path,
    datasets: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if not results_dir.exists():
        return []

    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Rapport : résultat illisible {path} : {exc}")
            continue
        if datasets is None or result.get("dataset") in datasets:
            results.append(result)
    return results


def _safe_pdf_text(value: Any) -> str:
    return str(value).replace("\u2013", "-").replace("\u2014", "-").encode(
        "latin-1", errors="replace"
    ).decode("latin-1")


def _format_metric(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/d"
    if not math.isfinite(number):
        return "n/d"
    return f"{number:.{decimals}f}"


def _rank_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    metric = config.REPORTING["ranking_metric"]
    empty_columns = [
        "dataset", "model", "status", metric,
        "val_mae", "test_mae", "test_rmse", "test_r2",
    ]
    completed = [
        row for row in results
        if row.get("status") == "completed" and row.get(metric) is not None
    ]
    frame = pd.DataFrame(completed)
    if frame.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(empty_columns)))

    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])
    if frame.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(empty_columns)))
    return frame.sort_values(
        ["dataset", metric],
        ascending=[
            True,
            config.REPORTING["ranking_direction"] == "minimize",
        ],
    ).reset_index(drop=True)


def _report_metric(row: Dict[str, Any]) -> float:
    try:
        value = float(row.get(config.REPORTING["ranking_metric"], float("inf")))
    except (TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def _plot_metric_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    metric = config.REPORTING["ranking_metric"]
    plot_data = frame.dropna(subset=[metric]).sort_values(metric)
    if plot_data.empty:
        return

    fig, axis = plt.subplots(figsize=(max(8, len(plot_data) * 1.1), 5))
    labels = [
        f"{row.dataset} / {row.model}"
        for row in plot_data.itertuples(index=False)
    ]
    axis.bar(labels, plot_data[metric], color="#668f80")
    axis.set_ylabel(metric)
    axis.set_title(f"Comparaison des modèles - {metric}")
    axis.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.REPORTING["graph_dpi"])
    plt.close(fig)


def _plot_test_predictions(
    results: Iterable[Dict[str, Any]],
    predictions_dir: Path,
    dataset_key: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    plotted = False
    limits = []
    for result in results:
        if result.get("status") != "completed":
            continue
        model_name = result.get("model")
        if not model_name:
            continue
        path = predictions_dir / f"{dataset_key}_{model_name}_predictions.csv"
        if not path.is_file():
            logger.warning(f"Prédictions absentes pour le rapport : {path}")
            continue
        predictions = pd.read_csv(path)
        test = predictions[predictions["split"] == "test"]
        if test.empty:
            continue
        actual = pd.to_numeric(test["actual"], errors="coerce").to_numpy()
        predicted = pd.to_numeric(test["predicted"], errors="coerce").to_numpy()
        valid = np.isfinite(actual) & np.isfinite(predicted)
        if not valid.any():
            continue
        axis.scatter(actual[valid], predicted[valid], label=model_name, alpha=0.8)
        limits.extend(actual[valid].tolist())
        limits.extend(predicted[valid].tolist())
        plotted = True

    if plotted:
        low, high = min(limits), max(limits)
        if low == high:
            low, high = low - 1, high + 1
        axis.plot([low, high], [low, high], linestyle="--", color="black")
        axis.set_xlabel("Rendement observé")
        axis.set_ylabel("Rendement prédit")
        axis.set_title(f"{dataset_key} - prédictions sur le test")
        axis.legend()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.REPORTING["graph_dpi"])
    plt.close(fig)


def _write_pdf_report(
    output_path: Path,
    title: str,
    results: List[Dict[str, Any]],
    metric_chart: Optional[Path] = None,
    prediction_chart: Optional[Path] = None,
) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _safe_pdf_text(title))
    pdf.ln(2)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(
        0,
        6,
        _safe_pdf_text(
            "Classement selon la métrique de validation configurée. "
            "Le jeu de test n'est pas utilisé pour le choix des hyperparamètres. "
            "Les résultats restent exploratoires et dépendent de la qualité, "
            "de la provenance et de la taille des données."
        ),
    )
    pdf.ln(3)

    completed = sorted(
        (row for row in results if row.get("status") == "completed"),
        key=_report_metric,
        reverse=config.REPORTING["ranking_direction"] != "minimize",
    )
    skipped = [row for row in results if row.get("status") != "completed"]

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, _safe_pdf_text("Modèles exécutés"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 8)
    headings = ("Modèle", "MAE val.", "MAE test", "RMSE test", "R2 test", "N test")
    widths = (54, 29, 29, 31, 26, 20)
    for heading, width in zip(headings, widths):
        pdf.cell(width, 7, _safe_pdf_text(heading), border=1)
    pdf.ln()

    pdf.set_font("Helvetica", size=8)
    if not completed:
        pdf.multi_cell(0, 6, _safe_pdf_text("Aucun modèle terminé pour cette sélection."))
    for row in completed:
        model_label = row.get("model", "n/d")
        if title.endswith("global"):
            model_label = f"{row.get('dataset', 'n/d')}/{model_label}"
        values = (
            model_label,
            _format_metric(row.get("val_mae")),
            _format_metric(row.get("test_mae")),
            _format_metric(row.get("test_rmse")),
            _format_metric(row.get("test_r2")),
            _format_metric(row.get("test_n"), decimals=0),
        )
        for value, width in zip(values, widths):
            pdf.cell(width, 7, _safe_pdf_text(value)[:28], border=1)
        pdf.ln()

    for row in completed:
        fallback_reason = row.get("fallback_reason")
        if fallback_reason:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                6,
                _safe_pdf_text(
                    f"{row.get('model', 'n/d')} (repli de "
                    f"{row.get('requested_model', 'TFT')}) : {fallback_reason}"
                ),
            )

    if skipped:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _safe_pdf_text("Modèles non exécutés"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        for row in skipped:
            details = (
                f"{row.get('model', 'n/d')} - {row.get('status', 'inconnu')}: "
                f"{row.get('reason', 'raison non précisée')}"
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6, _safe_pdf_text(details))

    for chart in (metric_chart, prediction_chart):
        if chart is not None and chart.is_file():
            pdf.add_page()
            pdf.image(str(chart), x=15, y=25, w=180)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def generate_phase7_reports(
    model_results_dir: Optional[Path] = None,
    predictions_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    graphs_dir: Optional[Path] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create ranked CSV, comparison plots, and per-dataset/global PDF reports."""
    model_results_dir = model_results_dir or (
        config.EXPERIMENTS_DIR / "model_results"
    )
    predictions_dir = predictions_dir or (
        config.EXPERIMENTS_DIR / "predictions"
    )
    reports_dir = reports_dir or config.REPORTS_DIR
    graphs_dir = graphs_dir or config.GRAPHS_DIR

    results = _read_model_results(model_results_dir, datasets)
    ranked = _rank_results(results)
    comparison_path = reports_dir / "BEST_MODELS_COMPARISON.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(comparison_path, index=False)

    selected_datasets = list(datasets) if datasets is not None else sorted({
        row.get("dataset") for row in results if row.get("dataset")
    })
    metric_charts: Dict[str, Path] = {}
    prediction_charts: Dict[str, Path] = {}

    if config.REPORTING["generate_graphs"]:
        global_chart = graphs_dir / "PHASE7_model_comparison.png"
        if not ranked.empty:
            _plot_metric_comparison(ranked, global_chart)
            metric_charts["global"] = global_chart

        for dataset_key in selected_datasets:
            dataset_rows = [row for row in results if row.get("dataset") == dataset_key]
            dataset_ranked = ranked[ranked["dataset"] == dataset_key] if not ranked.empty else ranked
            chart = graphs_dir / dataset_key / "PHASE7_model_comparison.png"
            if not dataset_ranked.empty:
                _plot_metric_comparison(dataset_ranked, chart)
                metric_charts[dataset_key] = chart

            prediction_chart = (
                graphs_dir / dataset_key / "PHASE7_test_predictions.png"
            )
            _plot_test_predictions(
                dataset_rows, predictions_dir, dataset_key, prediction_chart
            )
            if prediction_chart.is_file():
                prediction_charts[dataset_key] = prediction_chart

    for dataset_key in selected_datasets:
        dataset_rows = [
            row for row in results if row.get("dataset") == dataset_key
        ]
        _write_pdf_report(
            reports_dir / dataset_key / f"rapport_{dataset_key}.pdf",
            f"OliveSmart - Rapport {dataset_key}",
            dataset_rows,
            metric_charts.get(dataset_key),
            prediction_charts.get(dataset_key),
        )

    _write_pdf_report(
        reports_dir / "OLIVESMART_GLOBAL_REPORT.pdf",
        "OliveSmart - Rapport global",
        results,
        metric_charts.get("global"),
    )
    logger.info(
        f"Rapports phase 7 générés : {comparison_path} et {reports_dir}"
    )
    return {
        "n_results": len(results),
        "n_completed": len(ranked),
        "comparison_csv": comparison_path,
        "reports_dir": reports_dir,
        "metric_charts": metric_charts,
        "prediction_charts": prediction_charts,
    }
