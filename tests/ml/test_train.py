import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import joblib  # type: ignore[import-untyped]
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from typer.testing import CliRunner

from research_graph.cli import app
from research_graph.ml.prepare import SPLIT_FIELDS, read_csv, write_csv
from research_graph.ml.train import train


@pytest.fixture
def training_csv(tmp_path: Path) -> Path:
    examples = {
        "no_issue": "Excellent accessible app works perfectly",
        "low": "Minor font inconvenience slightly small text",
        "medium": "Small font makes reading difficult workaround magnification",
        "high": "Screen reader cannot access login completely inaccessible",
    }
    rows = [
        {
            "id": f"{label}-{i}",
            "group_id": f"{label}-{i}",
            "text": f"{text} example{i}",
            "label": label,
        }
        for label, text in examples.items()
        for i in range(3)
    ]
    source = tmp_path / "train.csv"
    write_csv(source, SPLIT_FIELDS, rows)
    return source


def test_training_bundle_roundtrip_and_no_holdout_vocabulary(
    training_csv: Path, tmp_path: Path
) -> None:
    # Reserved data lives next to train but must never be used by its vectorizer.
    write_csv(
        tmp_path / "test.csv",
        SPLIT_FIELDS,
        [
            {
                "id": "holdout",
                "group_id": "holdout",
                "text": "reservedholdoutword",
                "label": "high",
            }
        ],
    )
    output = tmp_path / "baseline-v1"
    metadata = train(training_csv, output)
    pipeline = joblib.load(output / "model.joblib")
    assert isinstance(pipeline, Pipeline)
    assert "reservedholdoutword" not in pipeline.named_steps["tfidf"].vocabulary_
    _, rows = read_csv(training_csv)
    texts = [row["text"] for row in rows]
    assert set(pipeline.predict(texts)) == {"no_issue", "low", "medium", "high"}
    probabilities = pipeline.predict_proba(texts)
    assert probabilities.shape == (12, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert metadata["training"] == {
        "source": str(training_csv),
        "sha256": hashlib.sha256(training_csv.read_bytes()).hexdigest(),
        "rows": 12,
        "groups": 12,
        "labels": {"no_issue": 3, "low": 3, "medium": 3, "high": 3},
        "seed": 42,
    }
    stored = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert stored["model_version"] == "baseline-v1"
    assert (
        stored["model_sha256"]
        == hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    )
    # A new interpreter loads the entire fitted pipeline without retraining.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import joblib,sys; print(joblib.load(sys.argv[1]).predict(['Excellent accessible app works perfectly'])[0])",
            str(output / "model.joblib"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "no_issue"
    second = tmp_path / "baseline-v2"
    train(training_csv, second)
    np.testing.assert_allclose(
        probabilities, joblib.load(second / "model.joblib").predict_proba(texts)
    )
    original = (output / "model.joblib").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        train(training_csv, output)
    assert (output / "model.joblib").read_bytes() == original


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("empty", "no rows"),
        ("unknown", "Unknown training label"),
        ("missing_class", "all four classes"),
        ("duplicate_id", "Duplicate training ID"),
        ("blank_text", "must not be empty"),
        ("group_conflict", "Conflicting training labels"),
        ("missing_columns", "requires id"),
        ("empty_vocabulary", "empty vocabulary"),
    ],
)
def test_invalid_training_data_has_no_artifacts(
    training_csv: Path, tmp_path: Path, mutation: str, message: str
) -> None:
    _, rows = read_csv(training_csv)
    if mutation == "empty":
        rows = []
    elif mutation == "unknown":
        rows[0]["label"] = "positive"
    elif mutation == "missing_class":
        rows = [row for row in rows if row["label"] != "low"]
    elif mutation == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "blank_text":
        rows[0]["text"] = "  "
    elif mutation == "group_conflict":
        rows[-1]["group_id"] = rows[0]["group_id"]
    elif mutation == "empty_vocabulary":
        for row in rows:
            row["text"] = "!!!"
    fields = ("text", "label") if mutation == "missing_columns" else SPLIT_FIELDS
    write_csv(training_csv, fields, rows)
    with pytest.raises(ValueError, match=message):
        train(training_csv, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_convergence_failure_has_no_artifacts(
    training_csv: Path, tmp_path: Path
) -> None:
    with (
        patch.object(
            Pipeline, "fit", side_effect=ConvergenceWarning("did not converge")
        ),
        pytest.raises(ValueError, match="increase --max-iter"),
    ):
        train(training_csv, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_train_cli(training_csv: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    args = [
        "ml",
        "train",
        "--source",
        str(training_csv),
        "--output",
        str(tmp_path / "model"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Model saved" in result.output
    assert runner.invoke(app, args).exit_code == 1
    assert runner.invoke(app, [*args, "--c", "0"]).exit_code == 2
    with pytest.raises(ValueError, match="C must be finite"):
        train(training_csv, tmp_path / "other", c=float("nan"))
