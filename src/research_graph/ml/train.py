"""Train and persist a TF-IDF / logistic regression baseline offline."""

import hashlib
import platform
import tempfile
import warnings
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib  # type: ignore[import-untyped]
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from research_graph.ml.prepare import LABELS, read_csv, write_json

CRITICALITY = {"no_issue": None, "low": 0, "medium": 1, "high": 2}


def train(
    source: Path,
    output: Path,
    seed: int = 42,
    c: float = 1.0,
    max_iter: int = 1000,
) -> dict[str, object]:
    """Fit exclusively on the provided training CSV; publish a complete bundle."""
    if output.exists():
        raise ValueError(
            "Output already exists; choose a new model version with --output"
        )
    if not 0 < c < float("inf") or max_iter < 1:
        raise ValueError("C must be finite and >0; max_iter must be >=1")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    fields, rows = read_csv(source)
    if not {"id", "group_id", "text", "label"} <= set(fields):
        raise ValueError("Training CSV requires id, group_id, text, label columns")
    if not rows:
        raise ValueError("Training CSV has no rows")
    ids: set[str] = set()
    group_labels: dict[str, str] = {}
    for row in rows:
        if any(not row[field].strip() for field in ("id", "group_id", "text")):
            raise ValueError("Training IDs, groups and texts must not be empty")
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown training label: {row['label']!r}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate training ID: {row['id']}")
        ids.add(row["id"])
        previous = group_labels.setdefault(row["group_id"], row["label"])
        if previous != row["label"]:
            raise ValueError(f"Conflicting training labels in group {row['group_id']}")
    counts = Counter(row["label"] for row in rows)
    if set(counts) != set(LABELS):
        raise ValueError(
            f"Training CSV must contain all four classes: {', '.join(LABELS)}"
        )

    vectorizer_params = {
        "ngram_range": (1, 2),
        "max_features": 50000,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }
    classifier_params = {
        "C": c,
        "max_iter": max_iter,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": seed,
    }
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("classifier", LogisticRegression(**classifier_params)),
        ]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit([row["text"] for row in rows], [row["label"] for row in rows])
    except ConvergenceWarning as exc:
        raise ValueError("Model did not converge; increase --max-iter") from exc

    metadata: dict[str, object] = {
        "model_version": output.name,
        "created_at": datetime.now(UTC).isoformat(),
        "algorithm": "TF-IDF + LogisticRegression",
        "language": "en",
        "classes": pipeline.classes_.tolist(),
        "criticality": CRITICALITY,
        "training": {
            "source": str(source),
            "sha256": source_hash,
            "rows": len(rows),
            "groups": len(group_labels),
            "labels": dict(counts),
            "seed": seed,
        },
        "parameters": {"tfidf": vectorizer_params, "classifier": classifier_params},
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "research-graph",
                    "scikit-learn",
                    "numpy",
                    "scipy",
                    "joblib",
                )
            },
        },
    }
    # Train before creating outputs; publish the model and metadata together.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".training-", dir=output.parent
    ) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        model = bundle / "model.joblib"
        joblib.dump(pipeline, model, compress=3)
        metadata["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        write_json(bundle / "metadata.json", metadata)
        bundle.rename(output)
    return metadata
