"""Offline ML commands."""

from pathlib import Path
from typing import Annotated

import typer

from research_graph.ml.annotate import annotate
from research_graph.ml.prepare import audit, prepare

app = typer.Typer(help="Audit, annotate and split ML data.", no_args_is_help=True)


@app.command("evaluate")
def evaluate_command(
    model_dir: Path = Path("artifacts/baseline-v1"),
    split_dir: Path = Path("data/processed/split-v1"),
    output: Path = Path("artifacts/baseline-v1/evaluation"),
) -> None:
    """Evaluate the saved model on validation/test and write metric reports."""
    from research_graph.ml.evaluate import evaluate

    try:
        report = evaluate(model_dir, split_dir, output)
    except (ValueError, TypeError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    for name, result in report["splits"].items():
        typer.echo(
            f"{name}: accuracy={result['accuracy']:.4f}; macro_f1={result['macro_f1']:.4f}"
        )
    typer.echo(f"Evaluation saved: {output}")


@app.command("train")
def train_command(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/processed/split-v1/train.csv"
    ),
    output: Path = Path("artifacts/baseline-v1"),
    seed: int = 42,
    c: Annotated[float, typer.Option(min=0.000001)] = 1.0,
    max_iter: Annotated[int, typer.Option(min=1)] = 1000,
) -> None:
    """Train TF-IDF + Logistic Regression and save a versioned model bundle."""
    # Other data/CLI commands do not need to import the training stack.
    from research_graph.ml.train import train

    try:
        metadata = train(source, output, seed, c, max_iter)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Model saved: {output / 'model.joblib'}; version: {metadata['model_version']}"
    )


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
