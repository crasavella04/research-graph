"""Проверка README: команды из блоков кода должны существовать (П5)."""

import re
import shlex
import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_graph.cli import app
from research_graph.config import Settings

README = Path(__file__).resolve().parents[1] / "README.md"
README_TEXT = README.read_text(encoding="utf-8")

SHELL_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)
ENV_ASSIGN_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=")
ENV_VAR_RE = re.compile(r"`(RG_[A-Z0-9_]+)`")

# Встроенные команды оболочки и coreutils: их нет как отдельных exe в Windows
SHELL_COMMANDS = {"cd", "cp"}

# Каталог с консольными скриптами окружения (.venv/Scripts или .venv/bin)
VENV_SCRIPTS = str(Path(sys.executable).parent)

runner = CliRunner()


def readme_commands() -> list[str]:
    commands = []
    for block in SHELL_BLOCK_RE.findall(README_TEXT):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line)
    return commands


def split_command(line: str) -> tuple[list[str], list[str]]:
    """Разделяет строку на присваивания переменных окружения и саму команду."""
    tokens = shlex.split(line)
    env = []
    while tokens and (match := ENV_ASSIGN_RE.match(tokens[0])):
        env.append(match.group(1))
        tokens.pop(0)
    return env, tokens


def cli_command_names() -> set[str]:
    names = set()
    for info in app.registered_commands:
        if info.name:
            names.add(info.name)
        elif info.callback:
            names.add(info.callback.__name__.replace("_", "-"))
    return names


COMMANDS = readme_commands()
SETTINGS_ENV_VARS = {f"RG_{name.upper()}" for name in Settings.model_fields}


def test_readme_has_commands() -> None:
    assert COMMANDS, "в README не найдено ни одного блока ```bash"


@pytest.mark.parametrize("line", COMMANDS)
def test_readme_command_exists(line: str) -> None:
    env, tokens = split_command(line)

    for var in env:
        assert var in SETTINGS_ENV_VARS, f"{var} нет среди настроек Settings"

    program, args = tokens[0], tokens[1:]
    if program in SHELL_COMMANDS:
        return
    assert shutil.which(program), f"программа {program!r} не найдена в PATH"

    if program == "uv" and args[:1] == ["run"]:
        tool, tool_args = args[1], args[2:]
        assert shutil.which(tool, path=VENV_SCRIPTS), (
            f"{tool!r} не установлен в окружение проекта"
        )
        if tool == "rg":
            # подкоманда должна существовать: `rg <cmd> --help` завершается успешно
            subcommand = [arg for arg in tool_args if not arg.startswith("-")][:1]
            result = runner.invoke(app, [*subcommand, "--help"])
            assert result.exit_code == 0, f"команды `rg {' '.join(subcommand)}` нет"


def test_all_cli_commands_documented() -> None:
    documented = {
        tokens[3]
        for _, tokens in map(split_command, COMMANDS)
        if tokens[:3] == ["uv", "run", "rg"] and len(tokens) > 3
    }
    missing = cli_command_names() - documented
    assert not missing, f"команды CLI не описаны в README: {sorted(missing)}"


def test_env_table_matches_settings() -> None:
    documented = set(ENV_VAR_RE.findall(README_TEXT))
    assert documented == SETTINGS_ENV_VARS, (
        f"нет в README: {sorted(SETTINGS_ENV_VARS - documented)}; "
        f"нет в Settings: {sorted(documented - SETTINGS_ENV_VARS)}"
    )
