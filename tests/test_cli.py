import pytest
from typer.testing import CliRunner

from research_graph.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_check_config_hides_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RG_GITHUB_TOKEN", "ghp_supersecretvalue")
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 0
    assert "ghp_supersecretvalue" not in result.stdout
