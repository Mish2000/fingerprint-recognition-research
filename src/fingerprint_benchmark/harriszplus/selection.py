"""Deterministic HarrisZ+ candidate filtering and uniform reselection."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Sequence, TypeVar

import numpy as np

from .types import SelectionCandidate


CandidateT = TypeVar("CandidateT")


def candidate_rank_key(candidate: Any) -> tuple[float, int, float, float, int]:
    """Frozen response/scale/location/source tie ordering."""

    response = float(candidate.response)
    if not math.isfinite(response):
        response = -math.inf
    return (
        -response,
        -int(candidate.scale_index),
        float(candidate.y),
        float(candidate.x),
        int(candidate.source_index),
    )


def rank_candidates(candidates: Iterable[CandidateT]) -> tuple[CandidateT, ...]:
    return tuple(sorted(candidates, key=candidate_rank_key))


def strict_local_maxima_numpy(
    response: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    tie_atol: float = 1.0e-6,
) -> np.ndarray:
    """Return candidates strictly greater than all 48 neighbors in a 7x7 window.

    At the image border the comparison uses the partial in-bounds
    neighborhood, matching same-sized dilation/max-pooling oracle behavior.
    """

    values = np.asarray(response, dtype=np.float32)
    mask = np.asarray(candidate_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("response and candidate_mask must be same-shaped 2D arrays.")
    if not math.isfinite(tie_atol) or tie_atol < 0.0:
        raise ValueError("tie_atol must be finite and non-negative.")
    import cv2

    footprint = np.ones((7, 7), dtype=np.uint8)
    pooled = cv2.dilate(
        values,
        footprint,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=-float("inf"),
    )
    # Raw response maps remain untouched.  Values within the frozen, tiny
    # float32 tolerance of the pooled maximum are treated as the same
    # mathematical maximum so backend roundoff cannot create pseudo-corners.
    is_maximum = np.isfinite(values) & ((pooled - values) <= tie_atol)
    local_maximum_count = cv2.boxFilter(
        is_maximum.astype(np.float32),
        ddepth=cv2.CV_32F,
        ksize=(7, 7),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    return mask & is_maximum & (local_maximum_count == 1.0)


def strict_local_maxima_torch(
    response: Any,
    candidate_mask: Any,
    *,
    tie_atol: float = 1.0e-6,
) -> Any:
    """Torch equivalent of :func:`strict_local_maxima_numpy`."""

    import torch

    if response.ndim != 2 or candidate_mask.shape != response.shape:
        raise ValueError("response and candidate_mask must be same-shaped 2D tensors.")
    if not math.isfinite(tie_atol) or tie_atol < 0.0:
        raise ValueError("tie_atol must be finite and non-negative.")
    import torch.nn.functional as functional

    pooled = functional.max_pool2d(
        response[None, None],
        kernel_size=7,
        stride=1,
        padding=3,
    )[0, 0]
    is_maximum = torch.isfinite(response) & ((pooled - response) <= tie_atol)
    local_maximum_count = functional.avg_pool2d(
        is_maximum.to(torch.float32)[None, None],
        kernel_size=7,
        stride=1,
        padding=3,
        count_include_pad=True,
        divisor_override=1,
    )[0, 0]
    return candidate_mask.to(torch.bool) & is_maximum & (local_maximum_count == 1.0)


def _has_neighbor_within(
    candidate: Any,
    buckets: dict[tuple[int, int], list[Any]],
    distance: float,
) -> bool:
    if distance <= 0.0:
        return False
    cell_x = math.floor(float(candidate.x) / distance)
    cell_y = math.floor(float(candidate.y) / distance)
    distance_squared = distance * distance
    for neighbor_y in range(cell_y - 1, cell_y + 2):
        for neighbor_x in range(cell_x - 1, cell_x + 2):
            for selected in buckets.get((neighbor_x, neighbor_y), ()):
                delta_x = float(candidate.x) - float(selected.x)
                delta_y = float(candidate.y) - float(selected.y)
                # The paper/oracle removes strictly closer candidates.  A
                # point exactly at the threshold survives.
                if delta_x * delta_x + delta_y * delta_y < distance_squared:
                    return True
    return False


def greedy_distance_suppression(
    candidates: Sequence[CandidateT],
    distance: float,
    *,
    already_ranked: bool = False,
    maximum_selected: int | None = None,
) -> tuple[tuple[CandidateT, ...], tuple[CandidateT, ...]]:
    """Greedily retain candidates separated by at least ``distance``."""

    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be finite and non-negative.")
    if maximum_selected is not None and maximum_selected <= 0:
        return (), tuple(candidates)
    ordered = tuple(candidates) if already_ranked else rank_candidates(candidates)
    selected: list[CandidateT] = []
    discarded: list[CandidateT] = []
    buckets: dict[tuple[int, int], list[CandidateT]] = defaultdict(list)
    for offset, candidate in enumerate(ordered):
        if maximum_selected is not None and len(selected) >= maximum_selected:
            discarded.extend(ordered[offset:])
            break
        if _has_neighbor_within(candidate, buckets, distance):
            discarded.append(candidate)
            continue
        selected.append(candidate)
        if distance > 0.0:
            cell = (
                math.floor(float(candidate.x) / distance),
                math.floor(float(candidate.y) / distance),
            )
            buckets[cell].append(candidate)
    return tuple(selected), tuple(discarded)


def scale_suppression_distance(working_sigma: float) -> int:
    if not math.isfinite(working_sigma) or working_sigma <= 0.0:
        raise ValueError("working_sigma must be finite and positive.")
    unrounded_distance = 3.0 * working_sigma
    nearest_integer = round(unrounded_distance)
    if math.isclose(unrounded_distance, nearest_integer, rel_tol=1.0e-12, abs_tol=1.0e-12):
        return int(nearest_integer)
    return int(math.ceil(unrounded_distance))


def parabolic_offset(left: float, center: float, right: float) -> float:
    """Unclipped 3-sample parabolic vertex offset from the center sample."""

    denominator = float(left) - 2.0 * float(center) + float(right)
    numerator = 0.5 * (float(left) - float(right))
    if denominator == 0.0 or not math.isfinite(denominator) or not math.isfinite(numerator):
        return 0.0
    offset = numerator / denominator
    return offset if math.isfinite(offset) else 0.0


def refine_subpixel_numpy(response: np.ndarray, x: int, y: int) -> tuple[float, float]:
    values = np.asarray(response, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("response must be two-dimensional.")
    center = float(values[y, x])
    offset_x = 0.0
    offset_y = 0.0
    if 0 < x < values.shape[1] - 1:
        offset_x = parabolic_offset(values[y, x - 1], center, values[y, x + 1])
    if 0 < y < values.shape[0] - 1:
        offset_y = parabolic_offset(values[y - 1, x], center, values[y + 1, x])
    return float(x) + offset_x, float(y) + offset_y


def refine_subpixel_torch(response: Any, x: Any, y: Any) -> tuple[Any, Any]:
    """Vectorized unclipped parabolic refinement for integer tensor indexes."""

    import torch

    center = response[y, x]
    x_left = torch.clamp(x - 1, min=0)
    x_right = torch.clamp(x + 1, max=response.shape[1] - 1)
    y_up = torch.clamp(y - 1, min=0)
    y_down = torch.clamp(y + 1, max=response.shape[0] - 1)
    denominator_x = response[y, x_left] - 2.0 * center + response[y, x_right]
    denominator_y = response[y_up, x] - 2.0 * center + response[y_down, x]
    numerator_x = 0.5 * (response[y, x_left] - response[y, x_right])
    numerator_y = 0.5 * (response[y_up, x] - response[y_down, x])
    valid_x = (x > 0) & (x < response.shape[1] - 1)
    valid_y = (y > 0) & (y < response.shape[0] - 1)
    valid_x &= torch.isfinite(denominator_x) & torch.isfinite(numerator_x) & (denominator_x != 0.0)
    valid_y &= torch.isfinite(denominator_y) & torch.isfinite(numerator_y) & (denominator_y != 0.0)
    offset_x = torch.where(valid_x, numerator_x / torch.where(valid_x, denominator_x, torch.ones_like(denominator_x)), torch.zeros_like(center))
    offset_y = torch.where(valid_y, numerator_y / torch.where(valid_y, denominator_y, torch.ones_like(denominator_y)), torch.zeros_like(center))
    return x.to(response.dtype) + offset_x, y.to(response.dtype) + offset_y


def eigen_axis_ratio(a_xx: float, a_xy: float, a_yy: float) -> float:
    """Return sqrt(lambda_min/lambda_max) for a symmetric 2x2 matrix."""

    a_xx = float(a_xx)
    a_xy = float(a_xy)
    a_yy = float(a_yy)
    if not all(math.isfinite(value) for value in (a_xx, a_xy, a_yy)):
        return 0.0
    trace = a_xx + a_yy
    discriminant_squared = (a_xx - a_yy) ** 2 + 4.0 * a_xy * a_xy
    if discriminant_squared < 0.0 or not math.isfinite(discriminant_squared):
        return 0.0
    discriminant = math.sqrt(discriminant_squared)
    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)
    if lambda_max <= 0.0 or lambda_min <= 0.0:
        return 0.0
    ratio = math.sqrt(lambda_min / lambda_max)
    return ratio if math.isfinite(ratio) else 0.0


def passes_eigen_axis_ratio(
    a_xx: float,
    a_xy: float,
    a_yy: float,
    *,
    threshold: float = 0.25,
) -> bool:
    """Strict HarrisZ+ eigen-axis-ratio test."""

    return eigen_axis_ratio(a_xx, a_xy, a_yy) > threshold


def remove_scale_01_duplicates(
    candidates: Sequence[CandidateT],
    *,
    distance: float = 1.0,
) -> tuple[CandidateT, ...]:
    """Suppress almost-duplicates only in the union of scale indexes 0 and 1."""

    duplicate_scales = [candidate for candidate in candidates if int(candidate.scale_index) in (0, 1)]
    other_scales = [candidate for candidate in candidates if int(candidate.scale_index) not in (0, 1)]
    retained, _ = greedy_distance_suppression(duplicate_scales, distance)
    return rank_candidates((*retained, *other_scales))


def uniform_selection_distance(height: int, width: int, maximum_keypoints: int) -> float:
    """Corrected Eq. 14: q = sqrt(8*m*n/(pi*k))."""

    if height <= 0 or width <= 0 or maximum_keypoints <= 0:
        raise ValueError("Image dimensions and maximum_keypoints must be positive.")
    return math.sqrt((8.0 * float(height) * float(width)) / (math.pi * maximum_keypoints))


def iterative_uniform_selection_with_diagnostics(
    candidates: Sequence[CandidateT],
    height: int,
    width: int,
    *,
    maximum_keypoints: int = 3000,
) -> tuple[tuple[CandidateT, ...], dict[str, Any]]:
    """Repeated pass-local greedy spacing with a strict output cap."""

    if maximum_keypoints <= 0 or maximum_keypoints > 3000:
        raise ValueError("maximum_keypoints must be in [1, 3000].")
    ordered = rank_candidates(candidates)
    distance = uniform_selection_distance(height, width, maximum_keypoints)
    if not ordered:
        return ordered, {
            "distance": distance,
            "passes": 0,
            "selected_per_pass": [],
            "input_count": len(ordered),
            "output_count": len(ordered),
            "cap_truncated_count": 0,
        }

    remaining: tuple[CandidateT, ...] = ordered
    output: list[CandidateT] = []
    selected_per_pass: list[int] = []
    while remaining and len(output) < maximum_keypoints:
        budget = maximum_keypoints - len(output)
        selected, discarded = greedy_distance_suppression(
            remaining,
            distance,
            already_ranked=True,
            maximum_selected=budget,
        )
        if not selected:
            # Defensive progress guarantee; a non-empty pass always accepts
            # its first candidate, but retain deterministic behavior if that
            # invariant is changed in the future.
            selected = (remaining[0],)
            discarded = remaining[1:]
        output.extend(selected)
        selected_per_pass.append(len(selected))
        remaining = discarded
    output_tuple = tuple(output[:maximum_keypoints])
    return output_tuple, {
        "distance": distance,
        "passes": len(selected_per_pass),
        "selected_per_pass": selected_per_pass,
        "input_count": len(ordered),
        "output_count": len(output_tuple),
        "cap_truncated_count": max(0, len(ordered) - len(output_tuple)),
    }


def iterative_uniform_selection(
    candidates: Sequence[CandidateT],
    height: int,
    width: int,
    *,
    maximum_keypoints: int = 3000,
) -> tuple[CandidateT, ...]:
    selected, _ = iterative_uniform_selection_with_diagnostics(
        candidates,
        height,
        width,
        maximum_keypoints=maximum_keypoints,
    )
    return selected


# Discoverable short names for mathematical unit tests and downstream code.
strict_local_maxima = strict_local_maxima_numpy
greedy_suppress = greedy_distance_suppression
subpixel_refine = refine_subpixel_numpy
uniform_select = iterative_uniform_selection
remove_duplicates = remove_scale_01_duplicates


__all__ = [
    "SelectionCandidate",
    "candidate_rank_key",
    "rank_candidates",
    "strict_local_maxima_numpy",
    "strict_local_maxima_torch",
    "greedy_distance_suppression",
    "scale_suppression_distance",
    "parabolic_offset",
    "refine_subpixel_numpy",
    "refine_subpixel_torch",
    "eigen_axis_ratio",
    "passes_eigen_axis_ratio",
    "remove_scale_01_duplicates",
    "uniform_selection_distance",
    "iterative_uniform_selection",
    "iterative_uniform_selection_with_diagnostics",
]
