"""Evaluate a saved model without fitting or changing any model parameters."""

import hashlib
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from research_graph.ml.prepare import (
    LABELS,
    duplicate_key,
    read_csv,
    write_csv,
    write_json,
)


def metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    """Use all four classes in macro averages, including zero-support classes."""
    if not actual or len(actual) != len(predicted):
        raise ValueError(
            "Expected non-empty, equally sized actual and predicted labels"
        )
    if (set(actual) | set(predicted)) - set(LABELS):
        raise ValueError("Metrics contain unknown labels")
    report = classification_report(
        actual, predicted, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(LABELS)
        ).tolist(),
        "label_order": list(LABELS),
    }


def load_splits(directory: Path) -> dict[str, list[dict[str, str]]]:
    """Reject changed/invalid data and ID, group or normalized-text leakage."""
    splits: dict[str, list[dict[str, str]]] = {}
    owners: dict[str, dict[str, str]] = {key: {} for key in ("id", "group_id", "text")}
    for name in ("train", "validation", "test"):
        fields, rows = read_csv(directory / f"{name}.csv")
        if not {"id", "group_id", "text", "label"} <= set(fields) or not rows:
            raise ValueError(
                f"{name}: requires non-empty CSV with id, group_id, text, label"
            )
        ids: set[str] = set()
        for row in rows:
            if row["label"] not in LABELS or any(
                not row[key].strip() for key in ("id", "group_id", "text")
            ):
                raise ValueError(f"{name}: empty fields or unknown label")
            if row["id"] in ids:
                raise ValueError(f"{name}: duplicate ID {row['id']}")
            ids.add(row["id"])
            for key, owner in owners.items():
                value = duplicate_key(row[key]) if key == "text" else row[key]
                previous = owner.setdefault(value, name)
                if previous != name:
                    raise ValueError(
                        f"Data leakage: overlapping {key} in {previous} and {name}"
                    )
        splits[name] = rows
    return splits


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation: {report['model_version']}",
        "",
        f"Data purpose: `{report['data_purpose']}`.",
        "Evaluation uses the provided labels. This run does not independently establish real-world criticality accuracy.",
        "",
        "Precision, recall and F1 below are macro averages over all four classes; undefined values use zero.",
        "",
        "| Split | Rows | Accuracy | Precision | Recall | Macro-F1 | Majority Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["splits"].items():
        lines.append(
            f"| {name} | {result['rows']} | {result['accuracy']:.4f} | "
            f"{result['precision_macro']:.4f} | {result['recall_macro']:.4f} | "
            f"{result['macro_f1']:.4f} | {result['majority_baseline']['macro_f1']:.4f} |"
        )
        if result["missing_classes"]:
            lines.append("")
            lines.append(
                f"Missing classes in {name}: {', '.join(result['missing_classes'])}."
            )
    for name, result in report["splits"].items():
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, scores in result["per_class"].items():
            lines.append(
                f"| {label} | {scores['precision']:.4f} | {scores['recall']:.4f} | "
                f"{scores['f1-score']:.4f} | {int(scores['support'])} |"
            )
        lines.extend(
            [
                "",
                "Confusion matrix: rows = actual, columns = predicted.",
                "",
                "| Actual / Predicted | " + " | ".join(LABELS) + " |",
                "|---|" + "---:|" * len(LABELS),
            ]
        )
        for label, values in zip(LABELS, result["confusion_matrix"], strict=True):
            lines.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines) + "\n"


def evaluate(model_dir: Path, split_dir: Path, output: Path) -> dict[str, Any]:
    """Publish metrics, matrices and row-level predictions for both holdouts."""
    if output.exists():
        raise ValueError("Output already exists; choose a new evaluation directory")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("training"), dict):
        raise TypeError("Model metadata must contain a training object")
    model_path = model_dir / "model.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != metadata.get("model_sha256"):
        raise ValueError("Model checksum does not match metadata")
    splits = load_splits(split_dir)
    hashes = {
        name: hashlib.sha256((split_dir / f"{name}.csv").read_bytes()).hexdigest()
        for name in splits
    }
    if hashes["train"] != metadata["training"].get("sha256"):
        raise ValueError("Training CSV does not match the model's training checksum")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or set(model.classes_) != set(LABELS):
        raise ValueError("Expected a fitted four-class Pipeline")
    manifest_path = split_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    if not isinstance(manifest, dict):
        raise TypeError("Split manifest must be a JSON object")
    majority = Counter(row["label"] for row in splits["train"]).most_common(1)[0][0]
    report: dict[str, Any] = {
        "model_version": metadata.get("model_version", model_dir.name),
        "created_at": datetime.now(UTC).isoformat(),
        "model_sha256": model_hash,
        "data_purpose": manifest.get("purpose", "unspecified"),
        "data_sha256": hashes,
        "majority_class_from_train": majority,
        "zero_division": 0,
        "splits": {},
    }
    predictions: dict[str, list[dict[str, str]]] = {}
    for name in ("validation", "test"):
        rows = splits[name]
        texts = [row["text"] for row in rows]
        actual = [row["label"] for row in rows]
        predicted = model.predict(texts).tolist()
        probabilities = model.predict_proba(texts)
        result = metrics(actual, predicted)
        result.update(
            rows=len(rows),
            missing_classes=sorted(set(LABELS) - set(actual)),
            majority_baseline=metrics(actual, [majority] * len(actual)),
        )
        report["splits"][name] = result
        predictions[name] = [
            {
                **row,
                "predicted_label": label,
                "confidence": str(
                    float(probabilities[i, list(model.classes_).index(label)])
                ),
                "correct": str(row["label"] == label).lower(),
            }
            for i, (row, label) in enumerate(zip(rows, predicted, strict=True))
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".evaluation-", dir=output.parent
    ) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        write_json(bundle / "report.json", report)
        (bundle / "report.md").write_text(markdown_report(report), encoding="utf-8")
        for name, records in predictions.items():
            write_csv(
                bundle / f"{name}_predictions.csv",
                (
                    "id",
                    "group_id",
                    "text",
                    "label",
                    "predicted_label",
                    "confidence",
                    "correct",
                ),
                records,
            )
            matrix = report["splits"][name]["confusion_matrix"]
            write_csv(
                bundle / f"{name}_confusion_matrix.csv",
                ("actual", *LABELS),
                [
                    {
                        "actual": label,
                        **{
                            col: str(value)
                            for col, value in zip(LABELS, values, strict=True)
                        },
                    }
                    for label, values in zip(LABELS, matrix, strict=True)
                ],
            )
        bundle.rename(output)
    return report
