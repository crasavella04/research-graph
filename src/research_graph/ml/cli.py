"""Offline ML commands."""

from pathlib import Path
from typing import Annotated

import typer

from research_graph.ml.annotate import annotate
from research_graph.ml.prepare import audit, prepare

app = typer.Typer(help="Audit, annotate and split ML data.", no_args_is_help=True)


@app.command("annotate")
def annotate_command(
    review: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Suggest weak labels, preserving human edits and backing up the CSV."""
    try:
        report = annotate(review)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Preliminary labels: {report['suggestions']}; human review required")


@app.command("audit")
def audit_command(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Path = Path("data/processed/audit-v1"),
    text_column: str = "review_text",
) -> None:
    """Create an audit report, cleaned data and a blank human review CSV."""
    try:
        report = audit(source, output, text_column)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Audit ready: {output}; unique texts: {report['unique_texts']}")


@app.command("prepare")
def prepare_command(
    review: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    audit_dir: Path = Path("data/processed/audit-v1"),
    output: Path = Path("data/processed/split-v1"),
    seed: int = 42,
) -> None:
    """Split only manually reviewed rows; fail if any class has <3 groups."""
    try:
        manifest = prepare(review, audit_dir, output, seed)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Splits ready: {output}; reviewed rows: {manifest['reviewed_rows']}")
