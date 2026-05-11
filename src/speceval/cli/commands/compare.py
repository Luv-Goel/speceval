"""``speceval compare`` — compare two evaluation runs."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def run(
    run_a: str = typer.Argument(
        ...,
        help="ID or path of the first (baseline) run.",
    ),
    run_b: str = typer.Argument(
        ...,
        help="ID or path of the second (comparison) run.",
    ),
    resamples: int = typer.Option(
        1000,
        "--resamples",
        "-n",
        help="Number of bootstrap resamples for p-value.",
    ),
) -> None:
    """Compare two evaluation runs by ID or path.

    Displays metric deltas, bootstrap p-values, and Cohen's d effect sizes.
    """
    from speceval.compare.delta import compare_runs
    from speceval.config import DEFAULT_STORE_PATH
    from speceval.store.sqlite import SQLiteStore

    store = SQLiteStore(DEFAULT_STORE_PATH)
    store.init_store()

    try:
        result = compare_runs(run_a, run_b, store, n_resamples=resamples)
    except Exception as exc:
        console.print(f"[red]✘[/red] Comparison failed: {exc}")
        raise typer.Exit(code=1)
    finally:
        store.close()

    # Print results
    console.print(f"[bold]Comparison:[/bold] {result.run_a} vs {result.run_b}")
    console.print()

    table = Table(title="Metric Deltas")
    table.add_column("Metric", style="cyan")
    table.add_column("Δ (B − A)", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Cohen's d", justify="right")
    table.add_column("Significance", justify="center")

    for metric in sorted(result.metric_deltas):
        d = result.metric_deltas[metric]
        p = result.significance[metric]
        es = result.effect_sizes[metric]
        sig = (
            "[green]p < 0.05[/green]"
            if p < 0.05
            else "[yellow]not sig.[/yellow]"
        )
        delta_str = f"{d:+.4f}"
        if d > 0:
            delta_str = f"[green]{delta_str}[/green]"
        elif d < 0:
            delta_str = f"[red]{delta_str}[/red]"

        table.add_row(metric, delta_str, f"{p:.4f}", f"{es:.3f}", sig)

    console.print(table)

    if result.details:
        console.print()
        console.print(f"Items in run A: {result.details['n_items_a']}")
        console.print(f"Items in run B: {result.details['n_items_b']}")
        console.print(f"Common metrics: {result.details['n_metrics']}")
