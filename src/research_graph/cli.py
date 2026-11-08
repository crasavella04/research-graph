import logging
from importlib.metadata import version as package_version

import typer
from pydantic import ValidationError

from research_graph.config import get_settings
from research_graph.logging_setup import setup_logging
from research_graph.ml.cli import app as ml_app

app = typer.Typer(help="research-graph command line interface.", no_args_is_help=True)
app.add_typer(ml_app, name="ml")

logger = logging.getLogger(__name__)


def mask_token(token: str | None) -> str:
    if not token:
        return "not set"
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}****{token[-4:]}"


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(package_version("research-graph"))


@app.command("check-config")
def check_config() -> None:
    """Load settings, set up logging and print a configuration summary."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        typer.echo(f"Invalid configuration:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc

    setup_logging(settings.log_level, settings.log_format)
    logger.debug("Settings loaded")

    typer.echo(f"github_token:    {mask_token(settings.github_token)}")
    typer.echo(f"data_dir:        {settings.data_dir.resolve()}")
    typer.echo(f"log_level:       {settings.log_level}")
    typer.echo(f"log_format:      {settings.log_format}")
    typer.echo(f"request_timeout: {settings.request_timeout}")
    typer.echo(f"max_concurrency: {settings.max_concurrency}")
