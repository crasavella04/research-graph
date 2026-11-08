import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_graph.cli import app
from research_graph.ml.prepare import (
    LABELS,
    REVIEW_FIELDS,
    audit,
    normalize,
    prepare,
    read_csv,
    write_csv,
)


def source(path: Path, texts: list[str]) -> Path:
    write_csv(
        path,
        ("review_text", "sentiment"),
        [{"review_text": text, "sentiment": "1"} for text in texts],
    )
    return path


def reviewed(tmp_path: Path) -> tuple[Path, Path]:
    # Distinct words ensure these synthetic reviews do not become similar groups.
    raw = source(tmp_path / "raw.csv", [f"review_{i}" for i in range(40)])
    directory = tmp_path / "audit"
    audit(raw, directory)
    _, rows = read_csv(directory / "review.csv")
    for i, row in enumerate(rows):
        row.update(label=LABELS[i % 4], reviewed="true", reviewer="test annotator")
    write_csv(directory / "review.csv", REVIEW_FIELDS, rows)
    return directory / "review.csv", directory


def test_audit_cleans_and_groups_without_creating_labels(tmp_path: Path) -> None:
    raw = source(
        tmp_path / "raw.csv",
        [
            "  Can't\nread the button! ",
            "can't read the button",
            " ",
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten eleven",
        ],
    )
    directory = tmp_path / "audit"
    report = audit(raw, directory)
    assert report["rows"] == 5
    assert report["empty_texts_removed"] == 1
    assert report["exact_duplicates_removed"] == 1
    assert report["unique_texts"] == 3
    assert report["similarity_groups"] == 2
    _, rows = read_csv(directory / "review.csv")
    assert all(row["label"] == "" and row["reviewed"] == "false" for row in rows)
    assert normalize("Can't\nread!") == "Can't read!"
    with pytest.raises(ValueError, match="already exists"):
        audit(raw, directory)


def test_reproducible_stratified_disjoint_splits(tmp_path: Path) -> None:
    review, directory = reviewed(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = prepare(review, directory, first)
    prepare(review, directory, second)
    group_sets = []
    for name in ("train", "validation", "test"):
        assert (first / f"{name}.csv").read_bytes() == (
            second / f"{name}.csv"
        ).read_bytes()
        _, rows = read_csv(first / f"{name}.csv")
        assert {row["label"] for row in rows} == set(LABELS)
        group_sets.append({row["group_id"] for row in rows})
    assert all(not a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1 :])
    assert manifest["reviewed_rows"] == 40
    with pytest.raises(ValueError, match="already exists"):
        prepare(review, directory, first)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("text", "modified", "changed"),
        ("group_id", "modified", "changed"),
        ("label", "positive", "valid label"),
        ("reviewer", "", "human reviewer"),
        ("reviewed", "yes", "true or false"),
    ],
)
def test_invalid_review_rejected(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    review, directory = reviewed(tmp_path)
    _, rows = read_csv(review)
    rows[0][field] = value
    write_csv(review, REVIEW_FIELDS, rows)
    with pytest.raises(ValueError, match=message):
        prepare(review, directory, tmp_path / "split")
    assert not (tmp_path / "split").exists()


def test_unreviewed_and_duplicate_ids(tmp_path: Path) -> None:
    review, directory = reviewed(tmp_path)
    _, rows = read_csv(review)
    rows[0]["reviewed"] = "false"
    write_csv(review, REVIEW_FIELDS, rows)
    result = prepare(review, directory, tmp_path / "split")
    assert result["excluded_unreviewed_rows"] == 1
    write_csv(review, REVIEW_FIELDS, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate review ID"):
        prepare(review, directory, tmp_path / "bad")


def test_conflicting_similar_reviews_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "audit"
    audit(
        source(
            tmp_path / "raw.csv",
            [
                "one two three four five six seven eight nine ten",
                "one two three four five six seven eight nine ten eleven",
            ],
        ),
        directory,
    )
    review = directory / "review.csv"
    _, rows = read_csv(review)
    for row, label in zip(rows, ("low", "high"), strict=True):
        row.update(label=label, reviewed="true", reviewer="human")
    write_csv(review, REVIEW_FIELDS, rows)
    with pytest.raises(ValueError, match="Conflicting labels"):
        prepare(review, directory, tmp_path / "split")


def test_cli_audit_and_pending_review(tmp_path: Path) -> None:
    raw = source(tmp_path / "raw.csv", ["example"])
    directory = tmp_path / "audit"
    runner = CliRunner()
    result = runner.invoke(
        app, ["ml", "audit", "--source", str(raw), "--output", str(directory)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads((directory / "audit.json").read_text())["rows"] == 1
    result = runner.invoke(
        app,
        [
            "ml",
            "prepare",
            "--review",
            str(directory / "review.csv"),
            "--audit-dir",
            str(directory),
        ],
    )
    assert result.exit_code == 1
    assert "needs >=3" in result.output
    result = runner.invoke(
        app, ["ml", "audit", "--source", str(raw), "--output", str(directory)]
    )
    assert result.exit_code == 1


def test_bad_csv(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("a,a\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique column"):
        read_csv(raw)
    raw.write_text("a,b\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        read_csv(raw)
    raw.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing text"):
        audit(raw, tmp_path / "audit")
    with (tmp_path / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["text"])
    with pytest.raises(ValueError, match="requires columns"):
        prepare(tmp_path / "review.csv", tmp_path, tmp_path / "split")
