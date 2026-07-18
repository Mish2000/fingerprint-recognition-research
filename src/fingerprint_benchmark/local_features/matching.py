"""L2 KNN matching, Lowe filtering, bidirectional union, and mutual consistency."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchRecord:
    index_a: int
    index_b: int
    distance: float
    second_distance: float


@dataclass(frozen=True)
class MatchSet:
    submitted: tuple[MatchRecord, ...]
    forward_ratio: tuple[MatchRecord, ...]
    reverse_ratio: tuple[MatchRecord, ...]
    diagnostics: dict[str, int | str]


def match_descriptors(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    *,
    lowe_ratio: float,
    matching_mode: str,
) -> MatchSet:
    forward_raw, forward = _ratio_direction(descriptors_a, descriptors_b, lowe_ratio, reverse=False)
    reverse_raw: int | None = None
    reverse: tuple[MatchRecord, ...] = ()
    if matching_mode != "one_way":
        reverse_raw, reverse = _ratio_direction(descriptors_b, descriptors_a, lowe_ratio, reverse=True)
    if matching_mode == "one_way":
        submitted = forward
    elif matching_mode == "bidirectional_union":
        submitted = _union(forward, reverse)
    elif matching_mode == "mutual":
        reverse_pairs = {(item.index_a, item.index_b) for item in reverse}
        submitted = tuple(item for item in forward if (item.index_a, item.index_b) in reverse_pairs)
    else:
        raise ValueError(f"Unsupported matching mode: {matching_mode!r}.")
    return MatchSet(
        submitted=submitted,
        forward_ratio=forward,
        reverse_ratio=reverse,
        diagnostics={
            "matching_mode": matching_mode,
            "raw_knn_count_a_to_b": int(forward_raw),
            "raw_knn_count_b_to_a": int(reverse_raw or 0),
            "ratio_match_count_a_to_b": len(forward),
            "ratio_match_count_b_to_a": len(reverse),
            "mutual_match_count": len(submitted) if matching_mode == "mutual" else 0,
            "matches_submitted_to_geometry": len(submitted),
        },
    )


def _ratio_direction(
    query: np.ndarray,
    train: np.ndarray,
    ratio: float,
    *,
    reverse: bool,
) -> tuple[int, tuple[MatchRecord, ...]]:
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(query, train, k=2)
    records: list[MatchRecord] = []
    for neighbors in knn:
        if len(neighbors) != 2:
            continue
        first, second = neighbors
        if float(first.distance) < float(ratio) * float(second.distance):
            index_a, index_b = (
                (int(first.trainIdx), int(first.queryIdx))
                if reverse
                else (int(first.queryIdx), int(first.trainIdx))
            )
            records.append(
                MatchRecord(index_a, index_b, float(first.distance), float(second.distance))
            )
    return len(knn), tuple(records)


def _union(
    first: tuple[MatchRecord, ...],
    second: tuple[MatchRecord, ...],
) -> tuple[MatchRecord, ...]:
    by_pair: dict[tuple[int, int], MatchRecord] = {}
    for record in (*first, *second):
        key = (record.index_a, record.index_b)
        current = by_pair.get(key)
        if current is None or record.distance < current.distance:
            by_pair[key] = record
    return tuple(by_pair[key] for key in sorted(by_pair))

__all__ = ["MatchRecord", "MatchSet", "match_descriptors"]
