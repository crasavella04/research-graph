# research-graph

[![CI](https://github.com/crasavella04/research-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/crasavella04/research-graph/actions/workflows/ci.yml)

`research-graph` — учебный проект курса «Разработка на Python и инженерия данных», который развивается в течение семестра.
Проект решает задачу анализа тональности пользовательских отзывов о доступности мобильных приложений: как люди с особыми потребностями оценивают удобство приложений и что им мешает.

Основа проекта — датасет [Accessibility User Reviews](https://www.kaggle.com/datasets/intelai/dataset-accessibility-user-reviews). Он содержит отзывы пользователей о доступности приложений с метками тональности.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git

## Установка

```bash
git clone https://github.com/crasavella04/research-graph.git
cd research-graph
uv sync
```

`uv sync` сам поставит нужную версию Python (из `.python-version`), создаст `.venv`
и установит зависимости строго по `uv.lock`.

Создайте локальный файл настроек из шаблона (необязательно — без него используются значения по умолчанию):

```bash
cp .env.example .env
```

## Конфигурация

Настройки читаются из переменных окружения с префиксом `RG_` или из файла `.env` в текущей директории.
Переменные окружения имеют приоритет над `.env`. Файл `.env` не коммитится.

| Переменная | Описание | По умолчанию | Ограничения |
|---|---|---|---|
| `RG_GITHUB_TOKEN` | Токен GitHub API. В выводе CLI всегда маскируется | не задан | — |
| `RG_DATA_DIR` | Директория для данных проекта | `data` |  |
| `RG_LOG_LEVEL` | Уровень логирования | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (регистр не важен) |
| `RG_LOG_FORMAT` | Формат логов: текст или одна JSON-строка на запись | `text` | `text`, `json` (регистр не важен) |
| `RG_REQUEST_TIMEOUT` | Таймаут HTTP-запросов, секунды | `10.0` | больше 0 и не больше 300 |
| `RG_MAX_CONCURRENCY` | Максимум одновременных запросов | `4` | от 1 до 64 |

## Использование

Список команд:

```bash
uv run rg --help
```

Версия пакета:

```bash
uv run rg version
```

```text
0.1.0
```

Проверка конфигурации — загружает настройки, настраивает логирование и печатает сводку:

```bash
uv run rg check-config
```

```text
github_token:    not set
data_dir:        /path/to/research-graph/data
log_level:       INFO
log_format:      text
request_timeout: 10.0
max_concurrency: 4
```

Если токен задан, он выводится только в маскированном виде, например `ghp_****abcd`.

Логи пишутся в stderr. Со структурным форматом каждая запись — одна JSON-строка,
которую удобно разбирать инструментами вроде `jq` или систем сбора логов:

```bash
RG_LOG_FORMAT=json RG_LOG_LEVEL=DEBUG uv run rg check-config
```

```text
{"timestamp": "2026-09-27T13:26:21.265711+00:00", "level": "DEBUG", "logger": "research_graph.cli", "message": "Settings loaded"}
github_token:    not set
...
```

Дополнительные поля передаются через `extra` и попадают в JSON как отдельные ключи:
`logger.info("loaded", extra={"rows": 1000})`.

При некорректном значении команда завершается с кодом 1 и объясняет, что не так:

```bash
RG_MAX_CONCURRENCY=0 uv run rg check-config
```

```text
Invalid configuration:
1 validation error for Settings
max_concurrency
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value='0', input_type=str]
```

## Разработка

Тесты с отчётом о покрытии:

```bash
uv run pytest
```

Линтер и проверка форматирования:

```bash
uv run ruff check .
uv run ruff format --check .
```

Проверка типов:

```bash
uv run mypy
```

Хуки pre-commit (ruff, ruff-format, mypy и базовые проверки) запускаются при каждом коммите.
Установка и ручной прогон по всем файлам:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Те же проверки выполняет CI в GitHub Actions на каждый push в `main` и pull request:
тесты гоняются на Python 3.12 и 3.13 под Ubuntu и Windows.

## Структура проекта

```text
research-graph/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI: ruff, mypy, pytest (матрица ОС и версий Python)
├── src/
│   └── research_graph/
│       ├── __init__.py
│       ├── py.typed
│       ├── cli.py              # Typer-приложение `rg`: команды проекта
│       ├── config.py           # настройки проекта
│       └── logging_setup.py    # настройка логирования: текст или JSON, приглушение библиотек
├── tests/
│   ├── conftest.py             # общие настройки для всех тестов
│   ├── test_cli.py             # тесты CLI через CliRunner
│   ├── test_config.py          # тесты настроек
│   └── test_logging.py         # тесты логирования и JSON-формата
├── .env.example                # шаблон настроек со всеми ключами и безопасными значениями
├── .gitignore
├── .pre-commit-config.yaml     # хуки pre-commit
├── .python-version             # версия Python для uv
├── pyproject.toml              # метаданные, зависимости, точка входа rg
├── uv.lock                     # зафиксированные версии зависимостей
└── README.md
```
