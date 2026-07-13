"""Deterministic hashing helpers for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
