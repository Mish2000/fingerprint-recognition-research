"""Candidate-directory publication for complete benchmark bundles."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


class BundlePublicationError(OSError):
    """Raised when a validated bundle cannot be published atomically."""


def create_candidate_directory(final_directory: Path) -> Path:
    parent = final_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            dir=parent,
            prefix=f".{final_directory.name}.candidate-",
        )
    )


def discard_candidate_directory(candidate_directory: Path) -> None:
    if candidate_directory.exists():
        shutil.rmtree(candidate_directory)


def publish_candidate_directory(candidate_directory: Path, final_directory: Path) -> None:
    """Promote one already-validated sibling directory, with rollback on error."""

    candidate = candidate_directory.resolve()
    final = final_directory.resolve()
    if candidate.parent != final.parent:
        raise BundlePublicationError("Candidate and final bundle directories must be siblings.")
    if not candidate.is_dir():
        raise BundlePublicationError(f"Candidate bundle directory does not exist: {candidate}")
    if final.exists():
        raise BundlePublicationError(
            f"Final bundle already exists and will not be overwritten: {final}"
        )

    try:
        os.replace(candidate, final)
    except OSError as exc:
        rollback_error: OSError | None = None
        # A mocked or unusual filesystem operation can move the directory and
        # still raise. Restore the pre-publication state whenever possible.
        if final.exists() and not candidate.exists():
            try:
                os.replace(final, candidate)
            except OSError as rollback_exc:
                rollback_error = rollback_exc
        message = f"Failed to publish benchmark bundle {final}: {exc}"
        if rollback_error is not None:
            message += f"; rollback also failed: {rollback_error}"
        raise BundlePublicationError(message) from exc
