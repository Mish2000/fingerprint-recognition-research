"""Small audit CLI for NIST SD300b/SD300c data discovery."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Iterable

from .canonical_fingers import (
    CANONICAL_FINGER_POSITIONS,
    canonical_finger_position,
    is_plain_multi_finger_capture,
)
from .nist_sd300 import DEFAULT_DATA_ROOT, ImageRecord, scan_all_datasets


def _canonical_counts(records: Iterable[ImageRecord], impression_type: str) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for record in records:
        if record.impression_type != impression_type:
            continue

        canonical_position = canonical_finger_position(record.impression_type, record.frgp)
        if canonical_position is not None:
            counts[canonical_position] += 1

    return {
        f"{position:02d}": counts[position]
        for position in CANONICAL_FINGER_POSITIONS
    }


def build_audit(data_root: Path) -> dict[str, object]:
    scan_results = scan_all_datasets(data_root)
    dataset_summaries = {
        dataset_name: result.audit_summary()
        for dataset_name, result in scan_results.items()
    }

    all_records = [
        record
        for result in scan_results.values()
        for record in result.records
    ]
    all_errors = [
        error
        for result in scan_results.values()
        for error in result.errors
    ]
    subjects = {record.subject_id for record in all_records}
    frgp_counts = Counter(record.frgp for record in all_records)
    finger_counts = Counter(record.finger_position for record in all_records)
    plain_multi_finger_captures = sum(
        1
        for record in all_records
        if is_plain_multi_finger_capture(record.impression_type, record.frgp)
    )

    totals = {
        "plain_images": sum(summary["plain_images"] for summary in dataset_summaries.values()),
        "roll_images": sum(summary["roll_images"] for summary in dataset_summaries.values()),
        "subjects": len(subjects),
        "finger_positions": dict(sorted(finger_counts.items())),
        "frgp": {f"{frgp:02d}": frgp_counts[frgp] for frgp in sorted(frgp_counts)},
        "canonical_single_finger_plain": _canonical_counts(all_records, "plain"),
        "canonical_single_finger_roll": _canonical_counts(all_records, "roll"),
        "plain_multi_finger_captures_not_pairable": plain_multi_finger_captures,
        "invalid_files": len(all_errors),
    }

    return {
        "data_root": str(data_root.resolve()),
        "datasets": dataset_summaries,
        "totals": totals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit NIST SD300b/SD300c image discovery without modifying datasets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root directory. Defaults to {DEFAULT_DATA_ROOT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for writing the JSON summary. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_audit(args.data_root)
    text = json.dumps(summary, indent=2, sort_keys=True)

    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
