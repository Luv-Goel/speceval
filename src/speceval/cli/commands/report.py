"""``speceval report`` — generate HTML reports."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def run(
    run_id: str = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Generate a single-run report for the given run ID.",
    ),
    compare: str = typer.Option(
        None,
        "--compare",
        "-c",
        help="Two run IDs separated by comma, e.g. ``run_a,run_b``, for a comparison report.",
    ),
    output: str = typer.Option(
        "./speceval_report.html",
        "--output",
        "-o",
        help="Output HTML file path.",
    ),
) -> None:
    """Generate an HTML report for a run or comparison of two runs."""
    from speceval.config import DEFAULT_STORE_PATH
    from speceval.store.sqlite import SQLiteStore

    store = SQLiteStore(DEFAULT_STORE_PATH)
    store.init_store()

    try:
        if compare:
            # Comparison report
            parts = [p.strip() for p in compare.split(",")]
            if len(parts) != 2:
                console.print("[red]✘[/red] --compare expects two run IDs separated by a comma.")
                raise typer.Exit(code=1)
            run_a, run_b = parts

            from speceval.compare.delta import compare_runs
            from speceval.report.html import generate_comparison_report

            comparison = compare_runs(run_a, run_b, store)
            out_path = generate_comparison_report(comparison, output)
            console.print(f"[green]✔[/green] Comparison report written to [bold]{out_path}[/bold]")

        elif run_id:
            # Single-run report
            from speceval.report.html import generate_html_report

            out_path = generate_html_report(run_id, store, output)
            console.print(
                f"[green]✔[/green] Report for run {run_id!r} written to"
                f" [bold]{out_path}[/bold]"
            )
        else:
            console.print("[red]✘[/red] Provide either --run-id or --compare.")
            raise typer.Exit(code=1)

    except Exception as exc:
        console.print(f"[red]✘[/red] Report generation failed: {exc}")
        raise typer.Exit(code=1)
    finally:
        store.close()
