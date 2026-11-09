"""Coordinate inference and synchronous training with short filesystem locks."""

import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from research_graph.backend.config import APISettings
from research_graph.backend.predictor import Predictor
from research_graph.backend.schemas import TrainingRequest
from research_graph.backend.storage import (
    DATA_FIELDS,
    FEEDBACK_FIELDS,
    Store,
    new_version,
    reviews,
)
from research_graph.ml.evaluate import evaluate
from research_graph.ml.prepare import (
    REVIEW_FIELDS,
    SPLIT_FIELDS,
    group_similar,
    prepare,
    read_csv,
    write_csv,
    write_json,
)
from research_graph.ml.train import train

logger = logging.getLogger(__name__)


class TrainingBusyError(Exception):
    pass


class ModelExistsError(Exception):
    pass


def remove_created(directory: Path, root: Path) -> None:
    """Only roll back newly published direct children of an owned version root."""
    if directory.resolve().parent != root.resolve():
        raise ValueError("Refusing cleanup outside the version root")
    if directory.exists():
        shutil.rmtree(directory)


class Backend:
    def __init__(self, data_dir: Path, settings: APISettings) -> None:
        self.settings = settings
        self.store = Store(data_dir, settings.artifacts_dir, settings.initial_dataset)
        self.lock = threading.RLock()
        self.training_lock = threading.Lock()
        self.predictor: Predictor | None = None
        candidates = self.store.model_directories()
        try:
            if candidates:
                self.predictor = Predictor(candidates[0])
        except Exception:
            logger.exception("Could not load the latest model")

    def health(self) -> dict[str, Any]:
        predictor = self.predictor
        return {
            "status": "ok" if predictor else "unavailable",
            "model_loaded": predictor is not None,
            "model_version": predictor.version if predictor else None,
            "training": self.training_lock.locked(),
        }

    def models(self) -> list[dict[str, Any]]:
        records = []
        for directory in self.store.model_directories():
            try:
                metadata = json.loads(
                    (directory / "metadata.json").read_text(encoding="utf-8")
                )
                report = directory / "evaluation" / "report.json"
                records.append(
                    {
                        "model_version": metadata["model_version"],
                        "active": bool(
                            self.predictor
                            and self.predictor.version == metadata["model_version"]
                        ),
                        "created_at": metadata.get("created_at"),
                        "dataset_version": metadata.get("dataset_version"),
                        "parameters": metadata.get("parameters", {}),
                        "training": metadata.get("training", {}),
                        "metrics": json.loads(report.read_text(encoding="utf-8"))[
                            "splits"
                        ]
                        if report.exists()
                        else None,
                    }
                )
            except (ValueError, KeyError, TypeError, OSError):
                logger.exception("Ignoring invalid model metadata: %s", directory)
        return records

    def load(self, model_bytes: bytes, metadata_bytes: bytes) -> dict[str, str]:
        with self.lock:
            if self.store.model_directories():
                raise ModelExistsError(
                    "A model already exists; /load is only for initial import"
                )
            with tempfile.TemporaryDirectory(
                prefix=".upload-", dir=self.store.artifacts
            ) as temporary:
                staged = Path(temporary) / "bundle"
                staged.mkdir()
                (staged / "model.joblib").write_bytes(model_bytes)
                (staged / "metadata.json").write_bytes(metadata_bytes)
                predictor = Predictor(staged)
                target = self.store.artifacts / predictor.version
                if target.exists():
                    raise ModelExistsError("Model version directory already exists")
                staged.rename(target)
                try:
                    self.store.commit(model_version=predictor.version)
                except Exception:
                    remove_created(target, self.store.artifacts)
                    raise
                self.predictor = predictor
        return {"status": "loaded", "model_version": predictor.version}

    def feedback(self, text: str, label: str) -> dict[str, Any]:
        with self.lock:
            return self.store.add_feedback(text, label)

    def retrain(self, options: TrainingRequest) -> dict[str, Any]:
        if not self.training_lock.acquire(blocking=False):
            raise TrainingBusyError("Another training request is already running")
        started = time.monotonic()
        try:
            return self._retrain(options, started)
        finally:
            self.training_lock.release()

    def _retrain(self, options: TrainingRequest, started: float) -> dict[str, Any]:
        with self.lock:
            parent = self.store.dataset()
            snapshot = self.store.feedback(parent)
            merged = {row["id"]: row for row in reviews(parent / "reviews.csv")}
            merged.update(
                {row["id"]: {key: row[key] for key in DATA_FIELDS} for row in snapshot}
            )
        dataset_version = new_version("dataset")
        model_version = new_version("model")
        final_data = self.store.datasets / dataset_version
        final_model = self.store.artifacts / model_version
        # Staging dirs are on the corresponding volume for atomic directory renames.
        with (
            tempfile.TemporaryDirectory(
                prefix=".training-", dir=self.store.datasets
            ) as data_temp,
            tempfile.TemporaryDirectory(
                prefix=".training-", dir=self.store.artifacts
            ) as model_temp,
        ):
            data = Path(data_temp) / dataset_version
            model = Path(model_temp) / model_version
            data.mkdir()
            records = sorted(merged.values(), key=lambda row: row["id"])
            write_csv(data / "reviews.csv", DATA_FIELDS, records)
            audit = data / "audit"
            audit.mkdir()
            group_similar(records)
            write_csv(audit / "cleaned.csv", SPLIT_FIELDS[:3], records)
            for row in records:
                row.update(reviewed="true", reviewer="dataset", notes="")
            write_csv(audit / "review.csv", REVIEW_FIELDS, records)
            write_json(
                audit / "audit.json",
                {"parent_dataset": parent.name, "feedback_rows": len(snapshot)},
            )
            splits = data / "splits"
            manifest = prepare(audit / "review.csv", audit, splits, options.seed)
            manifest["purpose"] = "user_supplied_labels"
            write_json(splits / "manifest.json", manifest)
            metadata = train(
                splits / "train.csv", model, options.seed, options.c, options.max_iter
            )
            report = evaluate(model, splits, model / "evaluation")
            metadata["dataset_version"] = dataset_version
            metadata["training"]["source"] = str(final_data / "splits" / "train.csv")  # type: ignore[index]
            write_json(model / "metadata.json", metadata)
            predictor = Predictor(model)
            predictions = {
                name: read_csv(model / "evaluation" / f"{name}_predictions.csv")[1]
                for name in ("validation", "test")
            }
            split_sizes = {
                name: len(read_csv(splits / f"{name}.csv")[1])
                for name in ("train", "validation", "test")
            }
            write_json(
                data / "metadata.json",
                {
                    "version": dataset_version,
                    "parent": parent.name,
                    "model_version": model_version,
                    "feedback_applied": len(snapshot),
                },
            )
            snapshot_revisions = {row["id"]: row["revision"] for row in snapshot}
            with self.lock:
                pending = [
                    row
                    for row in self.store.feedback(parent)
                    if snapshot_revisions.get(row["id"]) != row["revision"]
                ]
                write_csv(data / "feedback.csv", FEEDBACK_FIELDS, pending)
                created: list[tuple[Path, Path]] = []
                try:
                    data.rename(final_data)
                    created.append((final_data, self.store.datasets))
                    model.rename(final_model)
                    created.append((final_model, self.store.artifacts))
                    self.store.commit(
                        dataset_version=dataset_version, model_version=model_version
                    )
                except Exception:
                    for directory, root in reversed(created):
                        remove_created(directory, root)
                    raise
                self.predictor = predictor
            return {
                "status": "completed",
                "model_version": model_version,
                "dataset_version": dataset_version,
                "duration_seconds": round(time.monotonic() - started, 3),
                "split_sizes": split_sizes,
                "metrics": report["splits"],
                "predictions": predictions,
            }
