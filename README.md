# research-graph

`research-graph` — учебный проект курса «РРиИД Python», который развивается в течение семестра.
Задача проекта — собирать данные об исследовательских репозиториях (в том числе через GitHub API),
складывать их в локальное хранилище и строить по ним граф связей.

На текущем этапе (ЛР1) заложен фундамент: пакет в src-layout, типизированная конфигурация через
переменные окружения, единая настройка логирования, CLI `rg`, тесты, pre-commit и CI.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git

## Установка

```bash
git clone https://github.com/<user>/research-graph.git
cd research-graph
uv sync
```

`uv sync` сам поставит нужную версию Python (из `.python-version`), создаст `.venv`
и установит зависимости строго по `uv.lock`.

Создайте локальный файл настроек из шаблона (необязательно — без него используются значения по умолчанию):

```bash
cp .env.example .env
```

В PowerShell: `Copy-Item .env.example .env`.

## Конфигурация

Настройки читаются из переменных окружения с префиксом `RG_` или из файла `.env` в текущей директории.
Переменные окружения имеют приоритет над `.env`. Файл `.env` не коммитится.

| Переменная | Описание | По умолчанию | Ограничения |
|---|---|---|---|
| `RG_GITHUB_TOKEN` | Токен GitHub API. В выводе CLI всегда маскируется | не задан | — |
| `RG_DATA_DIR` | Директория для данных проекта | `data` | создаётся автоматически |
| `RG_LOG_LEVEL` | Уровень логирования | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (регистр не важен) |
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
request_timeout: 10.0
max_concurrency: 4
```

Если токен задан, он выводится только в маскированном виде, например `ghp_****abcd`.

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
uv run mypy src tests
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
│       ├── py.typed            # маркер PEP 561: пакет поставляется с аннотациями типов
│       ├── cli.py              # Typer-приложение `rg`: команды version и check-config
│       ├── config.py           # Settings (pydantic-settings), префикс RG_, get_settings()
│       └── logging_setup.py    # setup_logging(): формат, вывод в stderr, приглушение библиотек
├── tests/
│   ├── conftest.py             # автофикстура, изолирующая тесты от .env и переменных RG_*
│   ├── test_cli.py             # тесты CLI через CliRunner
│   └── test_config.py          # тесты настроек: значения по умолчанию, окружение, .env
├── .env.example                # шаблон настроек со всеми ключами и безопасными значениями
├── .gitignore
├── .pre-commit-config.yaml     # хуки pre-commit
├── .python-version             # версия Python для uv
├── pyproject.toml              # метаданные, зависимости, точка входа rg, настройки инструментов
├── uv.lock                     # зафиксированные версии зависимостей
└── README.md
```
