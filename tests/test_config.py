from pathlib import Path

import pytest
from pydantic import ValidationError

from research_graph.config import Settings, get_settings


def test_get_settings_returns_settings() -> None:
    assert isinstance(get_settings(), Settings)


def test_settings_defaults() -> None:
    settings = get_settings()

    assert settings.github_token is None
    assert settings.data_dir == Path("data")
    assert settings.log_level == "INFO"
    assert settings.request_timeout == 10.0
    assert settings.max_concurrency == 4


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RG_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RG_MAX_CONCURRENCY", "16")
    settings = Settings()
    assert settings.log_level == "DEBUG"
    assert settings.max_concurrency == 16


def test_all_fields_read_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RG_GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv("RG_DATA_DIR", str(tmp_path / "custom"))
    monkeypatch.setenv("RG_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("RG_REQUEST_TIMEOUT", "2.5")
    monkeypatch.setenv("RG_MAX_CONCURRENCY", "8")

    settings = get_settings()

    assert settings.github_token == "ghp_test"
    assert settings.data_dir == tmp_path / "custom"
    assert settings.log_level == "WARNING"
    assert settings.request_timeout == 2.5
    assert settings.max_concurrency == 8


def test_log_level_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RG_LOG_LEVEL", "debug")
    assert Settings().log_level == "DEBUG"


def test_env_without_prefix_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CONCURRENCY", "16")
    assert Settings().max_concurrency == 4


def test_settings_read_from_env_file() -> None:
    # isolated_env сделал chdir в tmp_path, поэтому .env создаётся во временной папке
    Path(".env").write_text(
        "RG_LOG_LEVEL=ERROR\nRG_REQUEST_TIMEOUT=30\n", encoding="utf-8"
    )

    settings = Settings()

    assert settings.log_level == "ERROR"
    assert settings.request_timeout == 30.0


def test_env_var_overrides_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text("RG_LOG_LEVEL=ERROR\n", encoding="utf-8")
    monkeypatch.setenv("RG_LOG_LEVEL", "DEBUG")
    assert Settings().log_level == "DEBUG"


def test_rejects_invalid_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RG_MAX_CONCURRENCY", "0")
    with pytest.raises(ValidationError):
        Settings()
