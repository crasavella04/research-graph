from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_graph.cli import app
from research_graph.ml.annotate import annotate, suggest
from research_graph.ml.prepare import REVIEW_FIELDS, read_csv, write_csv


@pytest.mark.parametrize(
    "text,label",
    [
        ("I cannot use this app with TalkBack.", "high"),
        ("The font is too small to read comfortably.", "medium"),
        ("The font is slightly too small.", "low"),
        ("Great app!", "no_issue"),
        ("Would love a dark theme", "no_issue"),
        ("I cannot log in because of a server error.", ""),
        ("The app crashes.", ""),
        ("TalkBack used to be broken but now works.", ""),
        ("Отличное приложение", ""),
    ],
)
def test_suggestions_are_conservative(text: str, label: str) -> None:
    assert suggest(text)[0] == label


def test_preserve_edits_and_never_mark_human_review(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    rows = [dict.fromkeys(REVIEW_FIELDS, "") for _ in range(3)]
    for row in rows:
        row.update(text="Great app", reviewed="false")
    rows[1].update(label="low", reviewer="human", reviewed="true")
    rows[2]["notes"] = "my note"
    write_csv(review, REVIEW_FIELDS, rows)
    original = review.read_bytes()
    report = annotate(review)
    _, result = read_csv(review)
    assert result[0]["label"] == "no_issue"
    assert result[0]["reviewed"] == "false"
    assert result[0]["reviewer"] == ""
    assert "AUTO" in result[0]["notes"]
    assert result[1:] == rows[1:]
    assert report["preserved_rows"] == 2
    assert review.with_name("review.before-rules-v1.csv").read_bytes() == original
    with pytest.raises(ValueError, match="Backup already exists"):
        annotate(review)


def test_annotate_cli(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    write_csv(review, REVIEW_FIELDS, [])
    runner = CliRunner()
    assert (
        runner.invoke(app, ["ml", "annotate", "--review", str(review)]).exit_code == 0
    )
    assert (
        runner.invoke(app, ["ml", "annotate", "--review", str(review)]).exit_code == 1
    )
