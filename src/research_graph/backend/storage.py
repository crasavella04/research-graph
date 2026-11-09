"""Atomic files and versioned datasets; callers serialize small mutations."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_graph.backend.predictor import safe_version
from research_graph.ml.prepare import (
    LABELS,
    digest,
    duplicate_key,
    normalize,
    read_csv,
    write_csv,
    write_json,
)

DATA_FIELDS = ("id", "text", "label")
FEEDBACK_FIELDS = (*DATA_FIELDS, "revision")


def new_version(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"


def atomic_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with tempfile.TemporaryDirectory(prefix=".write-", dir=path.parent) as temporary:
        staged = Path(temporary) / "data.csv"
        write_csv(staged, fields, rows)
        staged.replace(path)


def atomic_json(path: Path, value: object) -> None:
    with tempfile.TemporaryDirectory(prefix=".write-", dir=path.parent) as temporary:
        staged = Path(temporary) / "data.json"
        write_json(staged, value)
        staged.replace(path)


def reviews(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path)
    if not {"text", "label"} <= set(fields) or not rows:
        raise ValueError("Dataset must contain non-empty text and label columns")
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        text = normalize(row["text"])
        key = duplicate_key(text)
        if not key or row["label"] not in LABELS:
            raise ValueError("Dataset contains an empty text or invalid label")
        identifier = digest(key)
        if identifier in unique and unique[identifier]["label"] != row["label"]:
            raise ValueError("Dataset contains conflicting duplicate labels")
        unique[identifier] = {"id": identifier, "text": text, "label": row["label"]}
    return sorted(unique.values(), key=lambda row: row["id"])


class Store:
    def __init__(self, data_dir: Path, artifacts: Path, initial: Path | None) -> None:
        self.root = data_dir
        self.datasets = data_dir / "datasets"
        self.artifacts = artifacts
        self.initial = initial or data_dir / "processed" / "audit-v1" / "review.csv"
        self.root.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "backend_state.json"
        self.state: dict[str, Any] = (
            json.loads(self.state_file.read_text(encoding="utf-8"))
            if self.state_file.exists()
            else {}
        )

    def commit(self, **changes: str) -> None:
        state = {**self.state, **changes}
        atomic_json(self.state_file, state)
        self.state = state

    def dataset(self) -> Path:
        current = self.state.get("dataset_version")
        if current:
            return self.datasets / safe_version(current)
        if not self.initial.exists():
            raise FileNotFoundError(
                "Initial labeled dataset not found; configure RG_API_INITIAL_DATASET"
            )
        records = reviews(self.initial)
        version = new_version("dataset")
        target = self.datasets / version
        with tempfile.TemporaryDirectory(
            prefix=".initial-", dir=self.datasets
        ) as temporary:
            staged = Path(temporary) / version
            staged.mkdir()
            write_csv(staged / "reviews.csv", DATA_FIELDS, records)
            write_csv(staged / "feedback.csv", FEEDBACK_FIELDS, [])
            write_json(
                staged / "metadata.json",
                {"version": version, "source": str(self.initial)},
            )
            staged.rename(target)
        self.commit(dataset_version=version)
        return target

    def feedback(self, dataset: Path) -> list[dict[str, str]]:
        return read_csv(dataset / "feedback.csv")[1]

    def add_feedback(self, text: str, label: str) -> dict[str, Any]:
        dataset = self.dataset()
        rows = {row["id"]: row for row in self.feedback(dataset)}
        identifier = digest(duplicate_key(text))
        action = "updated" if identifier in rows else "added"
        rows[identifier] = {
            "id": identifier,
            "text": normalize(text),
            "label": label,
            "revision": uuid4().hex,
        }
        atomic_csv(
            dataset / "feedback.csv",
            FEEDBACK_FIELDS,
            sorted(rows.values(), key=lambda r: r["id"]),
        )
        return {
            "id": identifier,
            "label": label,
            "dataset_version": dataset.name,
            "action": action,
            "pending_count": len(rows),
        }

    def model_directories(self) -> list[Path]:
        candidates = [
            path.parent
            for path in self.artifacts.glob("*/metadata.json")
            if (path.parent / "model.joblib").exists()
            and not path.parent.name.startswith(".")
        ]
        return sorted(
            candidates,
            key=lambda path: (path / "metadata.json").stat().st_mtime_ns,
            reverse=True,
        )
