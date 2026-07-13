"""Result summary helpers for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import file_sha256
from .runner import read_result_rows, summarize_result_rows


def summarize_result_file(result_path: Path) -> dict[str, Any]:
    rows = read_result_rows(result_path)
    summary = summarize_result_rows(rows)
    summary["result_path"] = str(result_path)
    summary["result_sha256"] = file_sha256(result_path)
    return summary
