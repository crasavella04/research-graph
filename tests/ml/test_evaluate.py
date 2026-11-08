import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import joblib  # type: ignore[import-untyped]
import pytest
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from typer.testing import CliRunner

from research_graph.cli import app
from research_graph.ml.evaluate import evaluate, load_splits, metrics
from research_graph.ml.prepare import LABELS, SPLIT_FIELDS, read_csv, write_csv
from research_graph.ml.train import train


@pytest.fixture
def evaluation_data(tmp_path: Path) -> tuple[Path, Path]:
    splits = tmp_path / "splits"
    splits.mkdir()
    examples = {
        "no_issue": "Great app everything works perfectly",
        "low": "Slightly small fonts minor inconvenience",
        "medium": "Hard to read small text but zoom helps",
        "high": "Cannot log in using screen reader inaccessible",
    }
    for name in ("train", "validation", "test"):
        write_csv(
            splits / f"{name}.csv",
            SPLIT_FIELDS,
            [
                {
                    "id": f"{name}-{label}",
                    "group_id": f"{name}-{label}",
                    "text": f"{text} {name}",
                    "label": label,
                }
                for label, text in examples.items()
            ],
        )
    (splits / "manifest.json").write_text(
        '{"purpose":"functionality_check"}', encoding="utf-8"
    )
    model = tmp_path / "model"
    train(splits / "train.csv", model)
    return model, splits


def test_metrics_against_known_counts() -> None:
    result = metrics(
        ["no_issue", "no_issue", "low", "medium", "high"],
        ["no_issue", "medium", "no_issue", "medium", "high"],
    )
    assert result["accuracy"] == pytest.approx(0.6)
    assert result["precision_macro"] == pytest.approx(0.5)
    assert result["recall_macro"] == pytest.approx(0.625)
    assert result["macro_f1"] == pytest.approx((0.5 + 0 + 2 / 3 + 1) / 4)
    assert result["confusion_matrix"] == [
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert result["label_order"] == list(LABELS)
    assert metrics(["no_issue"], ["no_issue"])["macro_f1"] == 0.25
    with pytest.raises(ValueError, match="non-empty"):
        metrics([], [])
    with pytest.raises(ValueError, match="unknown labels"):
        metrics(["other"], ["low"])


def test_evaluation_is_read_only_and_writes_consistent_outputs(
    evaluation_data: tuple[Path, Path], tmp_path: Path
) -> None:
    model, splits = evaluation_data
    output = tmp_path / "evaluation"
    before = (model / "model.joblib").read_bytes()
    with patch.object(
        Pipeline, "fit", side_effect=AssertionError("evaluation must not fit")
    ):
        report = evaluate(model, splits, output)
    assert (model / "model.joblib").read_bytes() == before
    assert report["model_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["data_purpose"] == "functionality_check"
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert "rows = actual" in (output / "report.md").read_text(encoding="utf-8")
    pipeline = joblib.load(model / "model.joblib")
    for name in ("validation", "test"):
        result = report["splits"][name]
        assert result["rows"] == 4
        assert sum(map(sum, result["confusion_matrix"])) == 4
        _, predictions = read_csv(output / f"{name}_predictions.csv")
        actual = [row["label"] for row in predictions]
        predicted = [row["predicted_label"] for row in predictions]
        assert metrics(actual, predicted)["macro_f1"] == result["macro_f1"]
        probabilities = pipeline.predict_proba([row["text"] for row in predictions])
        for i, row in enumerate(predictions):
            assert float(row["confidence"]) == pytest.approx(
                probabilities[i, list(pipeline.classes_).index(row["predicted_label"])]
            )
        _, matrix = read_csv(output / f"{name}_confusion_matrix.csv")
        assert [[int(row[label]) for label in LABELS] for row in matrix] == result[
            "confusion_matrix"
        ]
        assert result["majority_baseline"]["accuracy"] == 0.25
    with pytest.raises(ValueError, match="already exists"):
        evaluate(model, splits, output)


@pytest.mark.parametrize("key", ["id", "group_id", "text"])
def test_leakage_is_rejected(
    evaluation_data: tuple[Path, Path], key: str, tmp_path: Path
) -> None:
    model, splits = evaluation_data
    _, training = read_csv(splits / "train.csv")
    _, test = read_csv(splits / "test.csv")
    test[0][key] = training[0][key]
    if key == "text":
        test[0][key] = test[0][key].upper() + "!!!"
    write_csv(splits / "test.csv", SPLIT_FIELDS, test)
    with pytest.raises(ValueError, match=f"overlapping {key}"):
        evaluate(model, splits, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def test_checksums_detect_changed_model_and_training_data(
    evaluation_data: tuple[Path, Path], tmp_path: Path
) -> None:
    model, splits = evaluation_data
    model_path = model / "model.joblib"
    original = model_path.read_bytes()
    model_path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Model checksum"):
        evaluate(model, splits, tmp_path / "evaluation")
    model_path.write_bytes(original)
    _, rows = read_csv(splits / "train.csv")
    rows[0]["text"] += " changed"
    write_csv(splits / "train.csv", SPLIT_FIELDS, rows)
    with pytest.raises(ValueError, match="Training CSV does not match"):
        evaluate(model, splits, tmp_path / "evaluation")


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("empty", "requires non-empty"),
        ("label", "unknown label"),
        ("duplicate", "duplicate ID"),
        ("blank", "empty fields"),
    ],
)
def test_bad_holdout_rejected(
    evaluation_data: tuple[Path, Path], mutation: str, message: str
) -> None:
    _, splits = evaluation_data
    _, rows = read_csv(splits / "test.csv")
    if mutation == "empty":
        rows = []
    elif mutation == "label":
        rows[0]["label"] = "positive"
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["text"] = " "
    write_csv(splits / "test.csv", SPLIT_FIELDS, rows)
    with pytest.raises(ValueError, match=message):
        load_splits(splits)


def test_evaluate_cli(evaluation_data: tuple[Path, Path], tmp_path: Path) -> None:
    model, splits = evaluation_data
    args = [
        "ml",
        "evaluate",
        "--model-dir",
        str(model),
        "--split-dir",
        str(splits),
        "--output",
        str(tmp_path / "evaluation"),
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "test: accuracy=" in result.output
    assert runner.invoke(app, args).exit_code == 1
