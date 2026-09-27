"""Main CLI entry point."""

from __future__ import annotations

import typer

from speceval.cli.commands import compare, init, report, run, validate

app = typer.Typer(
    name="speceval",
    help="Reproducible evaluation specifications for AI systems.",
    add_completion=False,
    no_args_is_help=True,
)

app.command(name="init")(init.run)
app.command(name="run")(run.run)
app.command(name="validate")(validate.run)
app.command(name="compare")(compare.run)
app.command(name="report")(report.run)


@app.command()
def list():
    """List recorded runs from the default store."""
    from speceval.config import DEFAULT_STORE_PATH
    from speceval.store.sqlite import SQLiteStore

    store = SQLiteStore(DEFAULT_STORE_PATH)
    store.init_store()
    runs = store.get_runs()
    if not runs:
        typer.echo("No runs found.")
        raise typer.Exit()

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Recent Runs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Dataset")
    table.add_column("Timestamp")
    table.add_column("Status")
    for r in runs:
        table.add_row(r["id"], r["model_name"], r["dataset_name"], r["timestamp"], r["status"])
    console.print(table)


if __name__ == "__main__":
    app()
