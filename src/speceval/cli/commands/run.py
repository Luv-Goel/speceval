"""``speceval run`` — execute an evaluation specification."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def run(
    spec: str = typer.Option(
        "speceval.yaml",
        "--spec",
        "-s",
        help="Path to the evaluation specification YAML file.",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results (overrides spec).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable result caching.",
    ),
    trials: int = typer.Option(
        None,
        "--trials",
        "-t",
        help="Override number of trials from spec.",
    ),
) -> None:
    """Load a spec, validate, run evaluation, store results, and print a summary.

    This is a skeleton that shows the intended workflow.  Full execution
    (model inference, dataset loading) is delegated to the engine.
    """
    spec_path = Path(spec)
    if not spec_path.exists():
        console.print(f"[red]✘[/red] Spec file not found: {spec}")
        raise typer.Exit(code=1)

    # Load spec
    try:
        import yaml
        raw = spec_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except Exception as exc:
        console.print(f"[red]✘[/red] Failed to load spec: {exc}")
        raise typer.Exit(code=1)

    # Quick validation
    if not isinstance(data, dict):
        console.print("[red]✘[/red] Spec must be a YAML mapping.")
        raise typer.Exit(code=1)
    for key in ("model", "dataset", "metrics"):
        if key not in data:
            console.print(f"[red]✘[/red] Missing required key: {key!r}")
            raise typer.Exit(code=1)

    model_name = data["model"].get("name", "unknown")
    dataset_name = data["dataset"].get("path", "unknown")
    metric_names = [m["name"] if isinstance(m, dict) else m for m in data.get("metrics", [])]

    # Overrides
    output_dir = Path(output) if output else Path(data.get("output_dir", "./speceval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_count = trials if trials is not None else data.get("trials", 1)

    # Generate run ID
    run_id = f"{model_name.replace('/', '_')}_{uuid.uuid4().hex[:8]}"

    # Store
    from speceval.config import DEFAULT_STORE_PATH
    from speceval.provenance.environment import capture_provenance
    from speceval.store.sqlite import SQLiteStore

    store = SQLiteStore(DEFAULT_STORE_PATH)
    store.init_store()

    # Capture provenance
    provenance = capture_provenance(cwd=spec_path.parent)

    # Register metrics
    from speceval.metrics import register_all
    register_all()

    # Save run metadata
    store.save_run(
        run_id=run_id,
        spec_hash=_hash_spec(raw),
        model_name=model_name,
        dataset_name=dataset_name,
        provenance_json=json.dumps(provenance),
        status="running",
    )

    # --- Simulated evaluation ---
    # In production this would load the dataset, run model inference,
    # compute metrics, and store results.
    console.print(f"[bold]Run ID:[/bold] {run_id}")
    console.print(f"[bold]Model:[/bold] {model_name}")
    console.print(f"[bold]Dataset:[/bold] {dataset_name}")
    console.print(f"[bold]Metrics:[/bold] {', '.join(metric_names)}")
    console.print(f"[bold]Trials:[/bold] {trial_count}")
    console.print()

    n_items = 5  # placeholder — would come from actual dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=n_items)

        for i in range(n_items):
            time.sleep(0.1)  # simulated inference
            prediction = f"simulated_output_{i}"
            expected = f"expected_output_{i}"

            # Compute metrics
            from speceval.metrics import compute_metric
            per_item_metrics = {}
            for mname in metric_names:
                try:
                    result = compute_metric(mname, [prediction], [expected])
                    per_item_metrics[mname] = result.value
                except Exception as exc:
                    console.print(f"  [yellow]⚠[/yellow] Metric {mname} failed: {exc}")

            store.save_result(
                run_id=run_id,
                item_index=i,
                input_json=json.dumps({"prompt": f"prompt_{i}"}),
                expected=expected,
                prediction=prediction,
                metrics_json=json.dumps(per_item_metrics),
                duration_ms=100.0,
            )
            progress.update(task, advance=1)

    # Mark run completed
    store.save_run(
        run_id=run_id,
        spec_hash=_hash_spec(raw),
        model_name=model_name,
        dataset_name=dataset_name,
        provenance_json=json.dumps(provenance),
        status="completed",
    )

    # Summary table
    results = store.get_results(run_id)
    console.print("\n[bold green]✔ Evaluation complete![/bold green]\n")

    if results:
        table = Table(title=f"Summary — {run_id}")
        table.add_column("Item", style="cyan")
        for mname in metric_names:
            table.add_column(mname.capitalize(), justify="right")
        table.add_column("Duration", justify="right")

        for item in results:
            ms = json.loads(item["metrics_json"])
            row = [str(item["item_index"])]
            for mname in metric_names:
                val = ms.get(mname, "—")
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            row.append(f"{item['duration_ms']:.0f} ms")
            table.add_row(*row)

        console.print(table)

    # Save output JSON
    out_file = output_dir / f"{run_id}_results.json"
    out_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": model_name,
                "dataset": dataset_name,
                "metrics": metric_names,
                "provenance": provenance,
                "results": [
                    {
                        "index": r["item_index"],
                        "metrics": json.loads(r["metrics_json"]),
                        "duration_ms": r["duration_ms"],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    console.print(f"\nResults saved to [bold]{out_file}[/bold]")

    store.close()


def _hash_spec(content: str) -> str:
    """Return a short hex digest of the spec content."""
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()[:12]
