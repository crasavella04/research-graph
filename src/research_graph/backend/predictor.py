"""Load a fitted pipeline once and use it for inference."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from research_graph.ml.prepare import LABELS
from research_graph.ml.train import CRITICALITY


def safe_version(version: object) -> str:
    if not isinstance(version, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", version
    ):
        raise ValueError("Invalid version name")
    return version


class Predictor:
    def __init__(self, directory: Path) -> None:
        self.metadata: dict[str, Any] = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        self.version = safe_version(self.metadata["model_version"])
        model_path = directory / "model.joblib"
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.metadata.get(
            "model_sha256"
        ):
            raise ValueError("Model checksum does not match metadata")
        self.pipeline = joblib.load(model_path)
        if not isinstance(self.pipeline, Pipeline) or set(
            self.pipeline.classes_
        ) != set(LABELS):
            raise ValueError("Expected a fitted four-class Pipeline")
        # Reject incompatible bundles before publishing them.
        self.predict("Model compatibility check")

    def predict(self, text: str) -> dict[str, Any]:
        label = str(self.pipeline.predict([text])[0])
        confidence = None
        if hasattr(self.pipeline, "predict_proba"):
            index = list(self.pipeline.classes_).index(label)
            probability = float(self.pipeline.predict_proba([text])[0, index])
            if math.isfinite(probability) and 0 <= probability <= 1:
                confidence = probability
        return {
            "criticality": CRITICALITY[label],
            "label": label,
            "confidence": confidence,
        }
