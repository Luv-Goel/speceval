"""HTML report generation using Jinja2 templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2

from speceval.compare.delta import ComparisonResult
from speceval.exceptions import SpecEvalError
from speceval.store.base import ResultStore

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    env.filters.setdefault("from_json", _from_json_filter)
    return env


def _from_json_filter(value: str) -> Any:
    """Jinja2 filter to parse a JSON string into a Python object."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def generate_html_report(
    run_id: str,
    store: ResultStore,
    output_path: str | Path,
) -> Path:
    """Generate a self-contained HTML report for a single run.

    Args:
        run_id: Run identifier to report on.
        store: Initialised result store containing the run data.
        output_path: Write destination (will be created if parent dirs exist).

    Returns:
        The resolved ``output_path``.

    Raises:
        SpecEvalError: If the run is not found or rendering fails.
    """
    run = store.get_run(run_id)
    if run is None:
        raise SpecEvalError(f"Run {run_id!r} not found in store")

    results = store.get_results(run_id)

    # Aggregate metrics across items
    aggregated = _aggregate_metrics(results)

    template = _env().get_template("report.html")
    html = template.render(
        title=f"Evaluation Report — {run_id}",
        run_id=run_id,
        run=run,
        provenance=json.loads(run.get("provenance_json", "{}")),
        aggregated_metrics=aggregated,
        results=results,
        total_items=len(results),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out.resolve()


def generate_comparison_report(
    comparison: ComparisonResult,
    output_path: str | Path,
) -> Path:
    """Generate an HTML report comparing two runs.

    Args:
        comparison: Result of ``compare_runs(...)``.
        output_path: Write destination.

    Returns:
        The resolved ``output_path``.
    """
    template = _env().get_template("report.html")
    html = template.render(
        title=f"Comparison — {comparison.run_a} vs {comparison.run_b}",
        comparison=comparison,
        metrics=sorted(comparison.metric_deltas.keys()),
        is_comparison=True,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out.resolve()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate per-item metrics into summary statistics.

    Returns a dict mapping metric name to ``{"mean": ..., "std": ..., "min": ..., "max": ...}``.
    """
    from statistics import mean, stdev

    raw: dict[str, list[float]] = {}
    for item in results:
        try:
            metrics = json.loads(item.get("metrics_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        for k, v in metrics.items():
            try:
                raw.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                continue

    aggregated: dict[str, dict[str, float]] = {}
    for name, vals in raw.items():
        if len(vals) == 0:
            continue
        aggregated[name] = {
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return aggregated


__all__ = ["generate_html_report", "generate_comparison_report"]
