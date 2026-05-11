"""``speceval validate`` — validate a specification file."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

console = Console()


def run(
    spec: str = typer.Option(
        "speceval.yaml",
        "--spec",
        "-s",
        help="Path to the evaluation specification YAML file.",
    ),
) -> None:
    """Load and validate an evaluation specification, printing warnings."""
    spec_path = Path(spec)

    if not spec_path.exists():
        console.print(f"[red]✘[/red] Spec file not found: {spec}")
        raise typer.Exit(code=1)

    try:
        import yaml
    except ImportError:
        console.print("[red]✘[/red] PyYAML is required. Install with: pip install pyyaml")
        raise typer.Exit(code=1)

    try:
        raw = spec_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        console.print(f"[red]✘[/red] YAML parse error: {exc}")
        raise typer.Exit(code=1)

    if not isinstance(data, dict):
        console.print("[red]✘[/red] Spec must be a YAML mapping (dictionary).")
        raise typer.Exit(code=1)

    warnings: list[str] = []
    errors: list[str] = []

    # Validate top-level keys
    required = ["model", "dataset", "metrics"]
    for key in required:
        if key not in data:
            errors.append(f"Missing required top-level key: {key!r}")

    if "spec_version" not in data:
        warnings.append("Missing 'spec_version' — assuming '1.0'")

    # Model section
    model = data.get("model", {})
    if isinstance(model, dict):
        if "provider" not in model:
            errors.append("model: missing 'provider'")
        if "name" not in model:
            errors.append("model: missing 'name'")
    else:
        errors.append("'model' must be a mapping")

    # Dataset section
    dataset = data.get("dataset", {})
    if isinstance(dataset, dict):
        if "source" not in dataset:
            errors.append("dataset: missing 'source'")
        if "path" not in dataset:
            errors.append("dataset: missing 'path'")
    else:
        errors.append("'dataset' must be a mapping")

    # Metrics section
    metrics = data.get("metrics", [])
    if not isinstance(metrics, list):
        errors.append("'metrics' must be a list")
    elif not metrics:
        warnings.append("No metrics specified")

    # Output
    if errors:
        console.print("[red]✘ Validation FAILED[/red]\n")
        for e in errors:
            console.print(f"  [red]•[/red] {escape(e)}")
        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for w in warnings:
                console.print(f"  [yellow]•[/yellow] {escape(w)}")
        raise typer.Exit(code=1)

    console.print("[green]✔[/green] Specification is valid!")
    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]•[/yellow] {escape(w)}")
    else:
        console.print("  No warnings.")
