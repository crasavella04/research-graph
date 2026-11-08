"""Auditable CSV preparation without inferring severity from sentiment."""

import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("no_issue", "low", "medium", "high")
REVIEW_FIELDS = ("id", "group_id", "text", "label", "reviewed", "reviewer", "notes")
SPLIT_FIELDS = ("id", "group_id", "text", "label")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """Keep punctuation and negation for the eventual text classifier."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def duplicate_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError("CSV must have unique column names")
        fields = list(reader.fieldnames)
        rows: list[dict[str, str]] = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record near line {line}")
            rows.append(row)
    return fields, rows


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def group_similar(rows: list[dict[str, str]]) -> None:
    """Group token-set Jaccard >= .85, transitively; short texts match exactly."""
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    index: dict[str, list[int]] = defaultdict(list)
    sets: list[set[str]] = []
    for i, row in enumerate(rows):
        tokens = set(duplicate_key(row["text"]).split())
        candidates: set[int] = set()
        if len(tokens) >= 5:
            for token in tokens:
                candidates.update(index[token])
            for j in sorted(candidates):
                other = sets[j]
                if (
                    len(other) >= 5
                    and len(tokens & other) / len(tokens | other) >= 0.85
                ):
                    parents[root(i)] = root(j)
            for token in tokens:
                index[token].append(i)
        sets.append(tokens)
    members: dict[int, list[str]] = defaultdict(list)
    for i, row in enumerate(rows):
        members[root(i)].append(row["id"])
    for i, row in enumerate(rows):
        row["group_id"] = min(members[root(i)])


def audit(
    source: Path, output: Path, text_column: str = "review_text"
) -> dict[str, object]:
    fields, original = read_csv(source)
    if text_column not in fields:
        raise ValueError(f"Missing text column: {text_column}")
    if output.exists():
        raise ValueError(
            "Output already exists; use a new directory to preserve reviews"
        )
    unique: dict[str, dict[str, str]] = {}
    empty = 0
    for row in original:
        text = normalize(row[text_column])
        key = duplicate_key(text)
        if not key:
            empty += 1
            continue
        identifier = digest(key)
        unique.setdefault(identifier, {"id": identifier, "text": text})
    records = sorted(unique.values(), key=lambda row: row["id"])
    group_similar(records)
    report: dict[str, object] = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "text_column": text_column,
        "rows": len(original),
        "columns": fields,
        "missing": {
            field: sum(not r[field].strip() for r in original) for field in fields
        },
        "existing_labels": {
            field: dict(Counter(r[field] for r in original))
            for field in ("is_accessibility", "sentiment")
            if field in fields
        },
        "empty_texts_removed": empty,
        "exact_duplicates_removed": len(original) - empty - len(records),
        "unique_texts": len(records),
        "similarity_groups": len({row["group_id"] for row in records}),
        "similarity_policy": "NFKC, casefold, word tokens; Jaccard >= 0.85 for >=5 unique tokens; transitive groups",
        "language": "not automatically verified; human language review required",
        "criticality_labels": "absent; sentiment and accessibility are not criticality",
    }
    output.mkdir(parents=True)
    write_csv(output / "cleaned.csv", SPLIT_FIELDS[:3], records)
    for row in records:
        row.update(label="", reviewed="false", reviewer="", notes="")
    write_csv(output / "review.csv", REVIEW_FIELDS, records)
    write_json(output / "audit.json", report)
    return report


def prepare(
    review: Path, audit_dir: Path, output: Path, seed: int = 42
) -> dict[str, object]:
    fields, rows = read_csv(review)
    if not set(REVIEW_FIELDS) <= set(fields):
        raise ValueError(f"Review file requires columns: {', '.join(REVIEW_FIELDS)}")
    _, cleaned = read_csv(audit_dir / "cleaned.csv")
    canonical = {row["id"]: row for row in cleaned}
    seen: set[str] = set()
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        identifier = row["id"]
        if identifier in seen:
            raise ValueError(f"Duplicate review ID: {identifier}")
        seen.add(identifier)
        source = canonical.get(identifier)
        if source is None or any(
            row[key] != source[key] for key in ("text", "group_id")
        ):
            raise ValueError(f"Review text/group changed or unknown ID: {identifier}")
        if row["reviewed"].strip().lower() not in ("true", "false"):
            raise ValueError("reviewed must be true or false")
        if row["reviewed"].strip().lower() == "false":
            continue
        row["label"] = row["label"].strip()
        if row["label"] not in LABELS or not row["reviewer"].strip():
            raise ValueError(
                "Reviewed rows require a valid label and a named human reviewer"
            )
        groups[row["group_id"]].append(row)
    by_label: dict[str, list[str]] = defaultdict(list)
    for group, members in groups.items():
        labels = {row["label"] for row in members}
        if len(labels) != 1:
            raise ValueError(
                f"Conflicting labels in similarity group {group}; resolve manually"
            )
        by_label[members[0]["label"]].append(group)
    for label in LABELS:
        if len(by_label[label]) < 3:
            raise ValueError(
                f"Class {label} needs >=3 reviewed independent groups; found {len(by_label[label])}"
            )
    if output.exists():
        raise ValueError("Output already exists; use a new version directory")
    splits: dict[str, list[dict[str, str]]] = {
        name: [] for name in ("train", "validation", "test")
    }
    for label in LABELS:
        ids = sorted(by_label[label])
        random.Random(f"{seed}:{label}").shuffle(ids)
        holdout = max(1, math.floor(len(ids) * 0.15))
        for name, selected in (
            ("test", ids[:holdout]),
            ("validation", ids[holdout : 2 * holdout]),
            ("train", ids[2 * holdout :]),
        ):
            for group in selected:
                splits[name].extend(groups[group])
    output.mkdir(parents=True)
    for name, records in splits.items():
        write_csv(
            output / f"{name}.csv", SPLIT_FIELDS, sorted(records, key=lambda r: r["id"])
        )
    manifest: dict[str, object] = {
        "seed": seed,
        "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
        "cleaned_sha256": hashlib.sha256(
            (audit_dir / "cleaned.csv").read_bytes()
        ).hexdigest(),
        "audit": json.loads((audit_dir / "audit.json").read_text(encoding="utf-8")),
        "policy": "per-class shuffled groups; floor(15%) validation/test, minimum one each; remainder train",
        "reviewed_rows": sum(len(records) for records in splits.values()),
        "excluded_unreviewed_rows": len(canonical)
        - sum(len(records) for records in splits.values()),
        "splits": {
            name: {
                "rows": len(records),
                "labels": dict(Counter(r["label"] for r in records)),
            }
            for name, records in splits.items()
        },
        "limitations": "Human review is required, not programmatically provable. Three groups/class is only a technical minimum. Freeze test before tuning. Similarity grouping is heuristic, not semantic deduplication.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
