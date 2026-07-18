"""Closed set of deterministic raw-score candidates."""

from __future__ import annotations

import math


def score_components(
    *,
    inliers: int,
    matches: int,
    keypoints_a: int,
    keypoints_b: int,
) -> dict[str, float]:
    inliers = int(inliers)
    matches = int(matches)
    ratio = float(inliers / matches) if matches > 0 else 0.0
    normalized = float(inliers / max(1, min(int(keypoints_a), int(keypoints_b))))
    composite = (
        float(inliers) * ratio * math.log1p(float(matches))
        if inliers > 0 and matches > 0
        else 0.0
    )
    return {
        "geometric_inlier_count": float(inliers),
        "geometric_inlier_ratio": ratio,
        "inliers_over_min_keypoints": normalized,
        "inliers_times_inlier_ratio_times_log1p_matches": composite,
    }


def raw_score(mode: str, components: dict[str, float]) -> float:
    if mode not in components:
        raise ValueError(f"Unsupported raw score mode: {mode!r}.")
    value = float(components[mode])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Raw local-feature score must be finite and non-negative.")
    return value

__all__ = ["raw_score", "score_components"]
