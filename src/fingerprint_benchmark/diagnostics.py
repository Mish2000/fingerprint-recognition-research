"""Deterministic score diagnostics for pairwise benchmark result rows.

The helpers in this module deliberately operate on result-row mappings rather
than importing the benchmark runner.  This keeps diagnostics usable for both
the historical v1 CSV files and contract-v2 result bundles.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .io import write_csv_atomic, write_json_atomic


OK_STATUS = "ok"
IDENTITY_FIELDS = ("protocol", "subject_id", "canonical_finger_position")
V1_SCORE_FORMAT = ".9g"

PAIRED_IDENTITY_COLUMNS = [
    "protocol",
    "subject_id",
    "canonical_finger_position",
    "sd300b_pair_id",
    "sd300c_pair_id",
    "sd300b_status",
    "sd300c_status",
    "sd300b_score",
    "sd300c_score",
    "delta_c_minus_b",
    "absolute_delta",
    "exact_equality",
    "both_zero",
]

V1_V2_COMPARISON_COLUMNS = [
    "pair_id",
    "v1_status",
    "v2_status",
    "v1_score",
    "v2_score",
    "v2_formatted_as_v1_9g",
    "delta_v2_minus_v1",
    "absolute_delta",
    "exact_numeric_equality",
    "explained_by_v1_9g",
    "classification",
]


class DiagnosticsError(ValueError):
    """Raised when rows cannot be diagnosed without ambiguous alignment."""


def score_diagnostics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic score diagnostics for one benchmark run.

    Counts for zero and positive scores, as well as score statistics, use only
    rows whose status is ``ok``.  Failure rows remain part of ``pair_count``
    and are summarized by status in ``failure_counts``.
    """

    materialized = list(rows)
    _index_by_pair_id(materialized, source="result rows")

    ok_scores: list[tuple[Mapping[str, Any], float]] = []
    failure_counts: Counter[str] = Counter()
    for row in materialized:
        status = _required_text(row, "status", context=_row_context(row))
        if status == OK_STATUS:
            ok_scores.append((row, _required_finite_score(row, context=_row_context(row))))
        else:
            failure_counts[status] += 1

    scores = [score for _, score in ok_scores]
    zero_rows = [(row, score) for row, score in ok_scores if score == 0.0]
    positions = Counter(
        _canonical_position(row, context=_row_context(row)) for row, _ in zero_rows
    )

    return {
        "pair_count": len(materialized),
        "ok_count": len(ok_scores),
        "failure_count": len(materialized) - len(ok_scores),
        "failure_counts": dict(sorted(failure_counts.items())),
        "zero_score_count": len(zero_rows),
        "positive_score_count": sum(score > 0.0 for score in scores),
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "mean": statistics.fmean(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "zero_score_pair_ids": sorted(
            _required_text(row, "pair_id", context=_row_context(row))
            for row, _ in zero_rows
        ),
        "zero_score_canonical_position_distribution": {
            position: positions[position]
            for position in sorted(positions, key=_position_sort_key)
        },
    }


def paired_sd300_diagnostics(
    sd300b_rows: Iterable[Mapping[str, Any]],
    sd300c_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare SD300b and SD300c scores by protocol/subject/finger identity.

    Alignment is always by ``(protocol, subject_id,
    canonical_finger_position)``.  Input row order is never used.  Summary
    score metrics use identities with an ``ok`` row in both datasets; shared
    identities containing a failure are retained in the detail records with
    null scores/deltas.
    """

    b_rows = list(sd300b_rows)
    c_rows = list(sd300c_rows)
    _validate_dataset(b_rows, expected="sd300b")
    _validate_dataset(c_rows, expected="sd300c")
    _index_by_pair_id(b_rows, source="SD300b rows")
    _index_by_pair_id(c_rows, source="SD300c rows")
    b_by_identity = _index_by_identity(b_rows, source="SD300b rows")
    c_by_identity = _index_by_identity(c_rows, source="SD300c rows")

    b_keys = set(b_by_identity)
    c_keys = set(c_by_identity)
    shared_keys = sorted(b_keys & c_keys, key=_identity_sort_key)
    b_only_keys = sorted(b_keys - c_keys, key=_identity_sort_key)
    c_only_keys = sorted(c_keys - b_keys, key=_identity_sort_key)

    details: list[dict[str, Any]] = []
    comparable_b: list[float] = []
    comparable_c: list[float] = []
    absolute_deltas: list[float] = []
    zero_overlap: list[dict[str, str]] = []
    exact_equality_count = 0

    for identity in shared_keys:
        b_row = b_by_identity[identity]
        c_row = c_by_identity[identity]
        b_status = _required_text(b_row, "status", context=_row_context(b_row))
        c_status = _required_text(c_row, "status", context=_row_context(c_row))
        b_score = (
            _required_finite_score(b_row, context=_row_context(b_row))
            if b_status == OK_STATUS
            else None
        )
        c_score = (
            _required_finite_score(c_row, context=_row_context(c_row))
            if c_status == OK_STATUS
            else None
        )

        comparable = b_score is not None and c_score is not None
        delta = c_score - b_score if comparable else None
        absolute_delta = abs(delta) if delta is not None else None
        exact_equality = b_score == c_score if comparable else None
        both_zero = b_score == 0.0 and c_score == 0.0 if comparable else None

        if comparable:
            comparable_b.append(b_score)
            comparable_c.append(c_score)
            absolute_deltas.append(absolute_delta)
            exact_equality_count += int(exact_equality)
            if both_zero:
                zero_overlap.append(_identity_dict(identity))

        details.append(
            {
                **_identity_dict(identity),
                "sd300b_pair_id": _required_text(
                    b_row, "pair_id", context=_row_context(b_row)
                ),
                "sd300c_pair_id": _required_text(
                    c_row, "pair_id", context=_row_context(c_row)
                ),
                "sd300b_status": b_status,
                "sd300c_status": c_status,
                "sd300b_score": b_score,
                "sd300c_score": c_score,
                "delta_c_minus_b": delta,
                "absolute_delta": absolute_delta,
                "exact_equality": exact_equality,
                "both_zero": both_zero,
            }
        )

    return {
        "alignment_key": list(IDENTITY_FIELDS),
        "sd300b_identity_count": len(b_by_identity),
        "sd300c_identity_count": len(c_by_identity),
        "shared_identity_count": len(shared_keys),
        "comparable_identity_count": len(comparable_b),
        "sd300b_only_identity_count": len(b_only_keys),
        "sd300c_only_identity_count": len(c_only_keys),
        "sd300b_only_identities": [_identity_dict(key) for key in b_only_keys],
        "sd300c_only_identities": [_identity_dict(key) for key in c_only_keys],
        "exact_equality_count": exact_equality_count,
        "mean_absolute_delta": (
            statistics.fmean(absolute_deltas) if absolute_deltas else None
        ),
        "median_absolute_delta": (
            statistics.median(absolute_deltas) if absolute_deltas else None
        ),
        "pearson_correlation": _pearson_correlation(comparable_b, comparable_c),
        "sd300b_zero_score_count": sum(score == 0.0 for score in comparable_b),
        "sd300c_zero_score_count": sum(score == 0.0 for score in comparable_c),
        "zero_overlap_count": len(zero_overlap),
        "zero_overlap_identities": zero_overlap,
        "identities": details,
    }


def compare_v1_v2_scores(
    v1_rows: Iterable[Mapping[str, Any]],
    v2_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare v1 and v2 raw scores after aligning rows by ``pair_id``.

    Historical v1 raw scores were serialized with ``.9g`` (nine significant
    digits).  Non-exact scores are classified as explained by old precision
    when converting the v2 value with ``format(value, '.9g')`` reproduces the
    numeric v1 value.  All other non-exact values are reported as differences
    beyond the old serialization precision.
    """

    old_rows = list(v1_rows)
    new_rows = list(v2_rows)
    old_by_pair = _index_by_pair_id(old_rows, source="v1 rows")
    new_by_pair = _index_by_pair_id(new_rows, source="v2 rows")

    old_ids = set(old_by_pair)
    new_ids = set(new_by_pair)
    shared_ids = sorted(old_ids & new_ids)
    v1_only_ids = sorted(old_ids - new_ids)
    v2_only_ids = sorted(new_ids - old_ids)

    details: list[dict[str, Any]] = []
    exact_count = 0
    explained_count = 0
    beyond_count = 0
    comparable_count = 0
    beyond_pair_ids: list[str] = []

    for pair_id in shared_ids:
        old_row = old_by_pair[pair_id]
        new_row = new_by_pair[pair_id]
        old_status = _required_text(old_row, "status", context=f"pair {pair_id!r}")
        new_status = _required_text(new_row, "status", context=f"pair {pair_id!r}")
        old_score = (
            _required_finite_score(old_row, context=f"v1 pair {pair_id!r}")
            if old_status == OK_STATUS
            else None
        )
        new_score = (
            _required_finite_score(new_row, context=f"v2 pair {pair_id!r}")
            if new_status == OK_STATUS
            else None
        )

        formatted_new: str | None = None
        delta: float | None = None
        absolute_delta: float | None = None
        exact: bool | None = None
        explained: bool | None = None
        classification = "not_comparable"

        if old_score is not None and new_score is not None:
            comparable_count += 1
            formatted_new = format(new_score, V1_SCORE_FORMAT)
            delta = new_score - old_score
            absolute_delta = abs(delta)
            exact = old_score == new_score
            if exact:
                explained = False
                classification = "exact_numeric_equality"
                exact_count += 1
            elif old_score == float(formatted_new):
                explained = True
                classification = "explained_by_v1_9g"
                explained_count += 1
            else:
                explained = False
                classification = "beyond_v1_precision"
                beyond_count += 1
                beyond_pair_ids.append(pair_id)

        details.append(
            {
                "pair_id": pair_id,
                "v1_status": old_status,
                "v2_status": new_status,
                "v1_score": old_score,
                "v2_score": new_score,
                "v2_formatted_as_v1_9g": formatted_new,
                "delta_v2_minus_v1": delta,
                "absolute_delta": absolute_delta,
                "exact_numeric_equality": exact,
                "explained_by_v1_9g": explained,
                "classification": classification,
            }
        )

    return {
        "alignment_key": "pair_id",
        "v1_raw_score_serialization": "format(float_value, '.9g')",
        "v1_significant_digits": 9,
        "v1_pair_count": len(old_by_pair),
        "v2_pair_count": len(new_by_pair),
        "shared_pair_count": len(shared_ids),
        "comparable_pair_count": comparable_count,
        "v1_only_pair_ids": v1_only_ids,
        "v2_only_pair_ids": v2_only_ids,
        "exact_numeric_equality_count": exact_count,
        "v1_9g_explained_difference_count": explained_count,
        "beyond_v1_precision_difference_count": beyond_count,
        "beyond_v1_precision_pair_ids": beyond_pair_ids,
        "pairs": details,
    }


def write_diagnostics_json(report: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write a diagnostics report with sorted JSON object keys."""

    write_json_atomic(dict(report), output_path)


def write_paired_diagnostics_csv(
    report: Mapping[str, Any], output_path: Path
) -> None:
    """Write paired identity detail rows in deterministic identity order."""

    raw_rows = report.get("identities")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise DiagnosticsError("Paired diagnostics report has no valid 'identities' list.")
    rows = [_csv_row(row, PAIRED_IDENTITY_COLUMNS) for row in raw_rows]
    rows.sort(
        key=lambda row: _identity_sort_key(
            (
                row["protocol"],
                row["subject_id"],
                row["canonical_finger_position"],
            )
        )
    )
    write_csv_atomic(rows, output_path, PAIRED_IDENTITY_COLUMNS)


def write_v1_v2_comparison_csv(
    report: Mapping[str, Any], output_path: Path
) -> None:
    """Write v1/v2 pair detail rows in deterministic pair-id order."""

    raw_rows = report.get("pairs")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise DiagnosticsError("v1/v2 diagnostics report has no valid 'pairs' list.")
    rows = [_csv_row(row, V1_V2_COMPARISON_COLUMNS) for row in raw_rows]
    rows.sort(key=lambda row: row["pair_id"])
    write_csv_atomic(rows, output_path, V1_V2_COMPARISON_COLUMNS)


def _index_by_pair_id(
    rows: Iterable[Mapping[str, Any]], *, source: str
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        pair_id = _required_text(row, "pair_id", context=source)
        if pair_id in index:
            raise DiagnosticsError(f"Duplicate pair_id {pair_id!r} in {source}.")
        index[pair_id] = row
    return index


def _index_by_identity(
    rows: Iterable[Mapping[str, Any]], *, source: str
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        identity = _identity(row)
        if identity in index:
            joined = "/".join(identity)
            raise DiagnosticsError(f"Duplicate paired identity {joined!r} in {source}.")
        index[identity] = row
    return index


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    context = _row_context(row)
    return (
        _required_text(row, "protocol", context=context),
        _required_text(row, "subject_id", context=context),
        _canonical_position(row, context=context),
    )


def _identity_dict(identity: tuple[str, str, str]) -> dict[str, str]:
    return dict(zip(IDENTITY_FIELDS, identity, strict=True))


def _identity_sort_key(identity: tuple[str, str, str]) -> tuple[Any, ...]:
    protocol, subject_id, position = identity
    return protocol, subject_id, _position_sort_key(position)


def _position_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def _canonical_position(row: Mapping[str, Any], *, context: str) -> str:
    value = _required_text(row, "canonical_finger_position", context=context)
    try:
        number = int(value)
    except ValueError:
        return value
    return str(number)


def _validate_dataset(rows: Iterable[Mapping[str, Any]], *, expected: str) -> None:
    for row in rows:
        actual = _required_text(row, "dataset", context=_row_context(row))
        if actual != expected:
            raise DiagnosticsError(
                f"Expected dataset {expected!r}, got {actual!r} in {_row_context(row)}."
            )


def _required_finite_score(row: Mapping[str, Any], *, context: str) -> float:
    raw_value = row.get("raw_score")
    if raw_value is None or str(raw_value).strip() == "":
        raise DiagnosticsError(f"Missing raw_score in {context}.")
    try:
        score = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticsError(f"Invalid raw_score {raw_value!r} in {context}.") from exc
    if not math.isfinite(score):
        raise DiagnosticsError(f"Non-finite raw_score {raw_value!r} in {context}.")
    return score


def _required_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise DiagnosticsError(f"Missing {field} in {context}.")
    return str(value)


def _row_context(row: Mapping[str, Any]) -> str:
    pair_id = row.get("pair_id")
    return f"pair {pair_id!r}" if pair_id not in (None, "") else "result row"


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Compute Pearson's r without a third-party dependency."""

    if len(xs) != len(ys):
        raise DiagnosticsError("Pearson inputs must have the same length.")
    if len(xs) < 2:
        return None
    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = math.fsum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_x)
        * math.fsum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return numerator / denominator


def _csv_row(row: Any, columns: Sequence[str]) -> dict[str, str]:
    if not isinstance(row, Mapping):
        raise DiagnosticsError("Diagnostics detail rows must be mappings.")
    missing = [column for column in columns if column not in row]
    if missing:
        raise DiagnosticsError(f"Diagnostics detail row is missing columns {missing}.")
    return {column: _csv_value(row[column]) for column in columns}


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)
