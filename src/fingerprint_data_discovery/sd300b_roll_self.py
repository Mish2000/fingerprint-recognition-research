"""Build and validate the SD300b roll_self protocol manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

from .nist_sd300 import DEFAULT_DATA_ROOT, ScanResult, scan_dataset
from .protocol_dataset import SD300B_CONTEXT
from .roll_self_protocol import (
    PROTOCOL,
    ManifestValidationReport,
    RollSelfPair,
    build_roll_self_pairs as build_context_roll_self_pairs,
    make_pair_id as make_context_pair_id,
    validate_manifest as validate_context_manifest,
    write_manifest_atomic as write_context_manifest_atomic,
)
from .self_manifest_common import write_validated_manifest


DATASET_CONTEXT = SD300B_CONTEXT
DATASET = DATASET_CONTEXT.name
DEFAULT_MANIFEST_PATH = DATASET_CONTEXT.manifest_path(PROTOCOL)


class ManifestGenerationError(ValueError):
    """Raised when the SD300b roll_self manifest cannot be generated safely."""


class ManifestValidationError(ValueError):
    """Raised when a roll_self manifest fails validation."""


def make_pair_id(subject_id: str, canonical_finger_position: int) -> str:
    return make_context_pair_id(DATASET_CONTEXT, subject_id, canonical_finger_position)


def build_roll_self_pairs(scan_result: ScanResult) -> list[RollSelfPair]:
    """Build deterministic SD300b roll_self pairs from a completed SD300b scan."""

    return build_context_roll_self_pairs(DATASET_CONTEXT, scan_result, ManifestGenerationError)


def generate_manifest(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_path: Path = DEFAULT_MANIFEST_PATH,
) -> ManifestValidationReport:
    """Scan SD300b, validate a candidate manifest, then atomically replace the target."""

    scan_result = scan_dataset(data_root, DATASET_CONTEXT.spec)
    pairs = build_roll_self_pairs(scan_result)
    return write_validated_manifest(
        pairs,
        output_path,
        lambda candidate_path: validate_manifest(candidate_path, data_root),
        ManifestGenerationError,
    )


def write_manifest_atomic(pairs: Iterable[RollSelfPair], output_path: Path) -> None:
    """Write the CSV manifest by replacing the target only after a full write succeeds."""

    write_context_manifest_atomic(pairs, output_path, ManifestGenerationError)


def validate_manifest(
    manifest_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> ManifestValidationReport:
    return validate_context_manifest(
        DATASET_CONTEXT,
        manifest_path,
        data_root,
        ManifestGenerationError,
        ManifestValidationError,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate the SD300b roll_self manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate and validate the manifest.")
    generate_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root directory. Defaults to {DEFAULT_DATA_ROOT}.",
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest path. Defaults to {DEFAULT_MANIFEST_PATH}.",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate an existing manifest.")
    validate_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root directory. Defaults to {DEFAULT_DATA_ROOT}.",
    )
    validate_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest path. Defaults to {DEFAULT_MANIFEST_PATH}.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "generate":
            report = generate_manifest(args.data_root, args.output)
            print(f"Wrote {args.output} with {report.row_count} validated pairs.")
        elif args.command == "validate":
            report = validate_manifest(args.manifest, args.data_root)
            print(f"Validated {args.manifest} with {report.row_count} pairs.")
        else:
            raise ManifestGenerationError(f"Unsupported command {args.command!r}.")
    except (ManifestGenerationError, ManifestValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
