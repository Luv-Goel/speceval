"""``speceval normalize`` — canonicalize a specification YAML file.

Reads a spec file, parses it into a :class:`SpecConfig`, and writes it back
in canonical form: keys in original order, ``None`` values stripped, and a
trailing newline. Useful for keeping checked-in specs tidy and diff-stable.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from speceval.spec.parse import parse_spec_string, spec_to_yaml

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
        help="Output path (overwrites *spec* when omitted).",
    ),
) -> None:
    """Canonicalize *spec*: strip None values, preserve key order, pretty-print YAML."""
    spec_path = Path(spec)
    if not spec_path.exists():
        console.print(f"[red]✘[/red] Spec file not found: {spec}")
        raise typer.Exit(code=1)

    try:
        raw = spec_path.read_text(encoding="utf-8")
        spec_obj = parse_spec_string(raw)
        canonical = spec_to_yaml(spec_obj)
    except Exception as exc:  # noqa: BLE001 — surface parse/validation errors to user
        console.print(f"[red]✘[/red] Failed to normalize spec: {exc}")
        raise typer.Exit(code=1)

    out_path = Path(output) if output else spec_path
    out_path.write_text(canonical, encoding="utf-8")
    console.print(f"[green]✔[/green] Normalized spec written to {out_path}")
