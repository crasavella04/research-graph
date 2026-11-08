"""Conservative English weak labels, always requiring human review."""

import re
import shutil
from collections import Counter
from pathlib import Path

from research_graph.ml.prepare import REVIEW_FIELDS, read_csv, write_csv, write_json

VERSION = "rules-v1"
ACCESSIBILITY = re.compile(
    r"\b(screen ?reader|talk ?back|voice ?over|accessib\w*|blind|deaf|"
    r"visually impaired|low vision|vision impair\w*|hearing impair\w*|"
    r"dyslex\w*|font\w*|text size|contrast|colou?r\s*blind\w*|"
    r"magnif\w*|zoom\w*|captions?|subtitles?|tts|text.to.speech)\b"
)
PROBLEM = re.compile(
    r"\b(can't|cannot|unable|doesn't|don't|won't|not|never|no|"
    r"small|tiny|hard|difficult|trouble|problem\w*|issue\w*|"
    r"unreadable|unusable|inaccessible|broken|crash\w*|fail\w*|"
    r"missing|lack\w*|annoy\w*|strain\w*|hurt\w*|but|however)\b"
)
BLOCKED = re.compile(
    r"\b(unusable|inaccessible|completely inaccessible|"
    r"(?:can't|cannot|unable to)\s+(?:\w+\s+){0,2}"
    r"(?:read|use|access|navigate|log\s*in|sign\s*in|login))\b"
)
PARTIAL = re.compile(
    r"\b(hard to|difficult to|trouble|too small|tiny|unreadable|"
    r"(?:doesn't|does not|not) work|not accessible|not compatible|"
    r"not read|doesn't read|doesn't announce|not announce|"
    r"missing|lack\w*|strain\w*|hurt\w*)\b"
)
MINOR = re.compile(r"\b(minor|slightly|a little|small inconvenience)\b")
PRAISE = re.compile(
    r"\b(great|excellent|awesome|perfect|amazing|love|wonderful|"
    r"fantastic|helpful|useful|good|works well|works as advertised|thank\w*)\b"
)
WISH = re.compile(r"\b(wish|would love|would like|please add|could you add)\b")


def suggest(text: str) -> tuple[str, str]:
    """Return a weak label plus an evidence note, or abstain."""
    text = text.casefold().replace("’", "'").replace("‘", "'")
    # Never silently turn ambiguous/foreign content into a negative example.
    if re.search(r"[\u0400-\u04ff\u4e00-\u9fff]", text):
        return "", "abstain: language requires manual review"
    sentences = re.split(r"[.!?;\n]+", text)
    candidates: list[tuple[int, str, str]] = []
    for sentence in sentences:
        if not ACCESSIBILITY.search(sentence):
            continue
        if re.search(r"\b(used to|fixed|resolved|no longer|now works)\b", sentence):
            continue  # Historical failures need a person to interpret their timeline.
        if BLOCKED.search(sentence):
            candidates.append((2, "high", sentence))
        elif PROBLEM.search(sentence) and MINOR.search(sentence):
            candidates.append((0, "low", sentence))
        elif PARTIAL.search(sentence):
            candidates.append((1, "medium", sentence))
    if candidates:
        _, label, evidence = max(candidates)
        return label, f"weak evidence (check context): {evidence.strip()}"
    if not PROBLEM.search(text) and (PRAISE.search(text) or WISH.search(text)):
        return (
            "no_issue",
            "weak evidence: praise/wish without a matched problem; verify full text",
        )
    return "", "abstain: consequences/accessibility context unclear; assign manually"


def annotate(review: Path) -> dict[str, object]:
    """Back up the input and fill only untouched rows; never claim human review."""
    fields, rows = read_csv(review)
    if not set(REVIEW_FIELDS) <= set(fields):
        raise ValueError("Review file is missing required columns")
    backup = review.with_name(f"{review.stem}.before-{VERSION}.csv")
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}; refusing to overwrite")
    counts: Counter[str] = Counter()
    preserved = 0
    for row in rows:
        if row["reviewed"].strip().lower() != "false" or any(
            row[field].strip() for field in ("label", "reviewer", "notes")
        ):
            preserved += 1
            continue
        label, reason = suggest(row["text"])
        row.update(
            label=label, notes=f"AUTO {VERSION}; {reason}; HUMAN REVIEW REQUIRED"
        )
        counts[label or "unresolved"] += 1
    shutil.copyfile(review, backup)
    # Write a temporary sibling first to avoid truncating a review on failure.
    temporary = review.with_suffix(".tmp")
    if temporary.exists():
        raise ValueError(f"Temporary file already exists: {temporary}")
    write_csv(temporary, tuple(fields), rows)
    temporary.replace(review)
    report: dict[str, object] = {
        "method": VERSION,
        "rows": len(rows),
        "suggestions": dict(counts),
        "preserved_rows": preserved,
        "backup": str(backup),
        "human_review": False,
        "limitations": "English regex weak labels, not model predictions or gold labels. No automatic class for ambiguous texts. No claim of annotation accuracy. All new rows remain reviewed=false and reviewer is blank.",
    }
    write_json(review.with_name("annotation_report.json"), report)
    return report
