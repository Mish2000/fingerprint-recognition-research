from __future__ import annotations

import math

import pytest

from fingerprint_benchmark.shared_accuracy_metrics import (
    calibrate_common_threshold,
    compute_operating_metrics,
    discrete_eer,
    exact_zero_false_accept_upper_bound,
    roc_det_points,
    trapezoidal_auc,
    wilson_interval,
)


def test_common_threshold_is_lowest_boundary_satisfying_b_and_c() -> None:
    calibration = calibrate_common_threshold(
        {
            "sd300c": [0.2, 0.5, 0.7, 0.95],
            "sd300b": [0.1, 0.4, 0.8, 0.9, None],
        },
        0.25,
    )

    assert calibration.threshold == pytest.approx(0.9)
    assert list(calibration.per_dataset) == ["sd300b", "sd300c"]
    assert calibration.per_dataset["sd300b"].far == pytest.approx(0.25)
    assert calibration.per_dataset["sd300c"].far == pytest.approx(0.25)
    assert calibration.per_dataset["sd300b"].failure_count == 1
    assert calibration.per_dataset["sd300b"].scores_at_threshold == 1


def test_integer_ties_require_the_explicit_max_plus_one_candidate() -> None:
    calibration = calibrate_common_threshold(
        {
            "sd300b": [0, 4, 4],
            "sd300c": [0, 1, 4],
        },
        0.34,
        integer_scores=True,
    )

    assert calibration.threshold == 5
    assert calibration.integer_scores is True
    assert calibration.per_dataset["sd300b"].false_accept_count == 0
    assert calibration.per_dataset["sd300c"].false_accept_count == 0
    assert calibration.per_dataset["sd300b"].scores_at_threshold == 0


def test_integer_score_gap_selects_the_lowest_unobserved_integer_boundary() -> None:
    calibration = calibrate_common_threshold(
        {
            "sd300b": [0, 4],
            "sd300c": [0, 4],
        },
        0.5,
        integer_scores=True,
    )

    # Thresholds 1 through 4 produce the same decisions.  Integer calibration
    # must choose the lowest valid boundary, even though score 1 is unobserved.
    assert calibration.threshold == 1
    assert calibration.candidate_count == 6
    assert calibration.per_dataset["sd300b"].far == 0.5
    assert calibration.per_dataset["sd300c"].far == 0.5
    assert calibration.per_dataset["sd300b"].scores_at_threshold == 0
    assert calibration.per_dataset["sd300c"].scores_at_threshold == 0


def test_operating_metrics_use_inclusive_threshold_and_separate_failures() -> None:
    metrics = compute_operating_metrics(
        genuine_scores=[9, 5, 4, None],
        impostor_scores=[5, 2, None],
        threshold=5,
        integer_scores=True,
    )

    assert metrics.genuine_total_count == 4
    assert metrics.genuine_scored_count == 3
    assert metrics.genuine_failure_count == 1
    assert metrics.impostor_total_count == 3
    assert metrics.impostor_scored_count == 2
    assert metrics.impostor_failure_count == 1
    assert metrics.true_accept_count == 2
    assert metrics.false_nonmatch_count == 1
    assert metrics.false_accept_count == 1
    assert metrics.true_nonmatch_count == 1
    assert metrics.genuine_scores_at_threshold == 1
    assert metrics.impostor_scores_at_threshold == 1
    assert metrics.tar == pytest.approx(2 / 3)
    assert metrics.fnmr == pytest.approx(1 / 3)
    assert metrics.far == pytest.approx(1 / 2)
    assert metrics.zero_false_accept_upper_95 is None


def test_zero_false_accept_operating_point_reports_exact_upper_bound() -> None:
    metrics = compute_operating_metrics(
        genuine_scores=[2],
        impostor_scores=[0] * 100,
        threshold=1,
        integer_scores=True,
    )

    assert metrics.far == 0.0
    assert metrics.far_wilson_95 is not None
    assert metrics.far_wilson_95.lower == 0.0
    assert metrics.far_wilson_95.upper == pytest.approx(0.0369934982)
    assert metrics.zero_false_accept_upper_95 == pytest.approx(0.0295130496)


def test_wilson_and_exact_zero_bounds_validate_and_match_known_values() -> None:
    interval = wilson_interval(5, 100)

    assert interval.confidence == 0.95
    assert interval.lower == pytest.approx(0.0215436792)
    assert interval.upper == pytest.approx(0.1117504692)
    assert exact_zero_false_accept_upper_bound(100) == pytest.approx(0.0295130496)

    with pytest.raises(ValueError, match="positive integer"):
        exact_zero_false_accept_upper_bound(0)
    with pytest.raises(ValueError, match="trial_count"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(0, 100, confidence=1.0)


def test_unsmoothed_roc_det_auc_and_discrete_eer_are_deterministic() -> None:
    first = roc_det_points(
        genuine_scores=[2, 1, None],
        impostor_scores=[1, 0, None],
        integer_scores=True,
    )
    second = roc_det_points(
        genuine_scores=[2, 1, None],
        impostor_scores=[1, 0, None],
        integer_scores=True,
    )

    assert first == second
    assert [point.threshold for point in first] == [3, 2, 1, 0]
    assert [(point.far, point.tar, point.fnmr) for point in first] == [
        (0.0, 0.0, 1.0),
        (0.0, 0.5, 0.5),
        (0.5, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    assert first[2].genuine_scores_at_threshold == 1
    assert first[2].impostor_scores_at_threshold == 1
    assert all(point.genuine_failure_count == 1 for point in first)
    assert all(point.impostor_failure_count == 1 for point in first)
    assert trapezoidal_auc(first) == pytest.approx(0.875)

    eer = discrete_eer(first)
    assert eer.threshold == 2
    assert eer.eer == pytest.approx(0.25)
    assert eer.far == 0.0
    assert eer.fnmr == 0.5
    assert eer.absolute_gap == 0.5


def test_large_unique_continuous_roc_has_exact_incremental_counts_and_endpoints() -> None:
    sample_count = 5_000
    genuine = [float(2 * index + 1) for index in range(sample_count)]
    impostors = [float(2 * index) for index in range(sample_count)]

    points = roc_det_points(genuine, impostors)

    assert len(points) == 2 * sample_count + 1
    assert points[0].threshold > genuine[-1]
    assert (
        points[0].true_accept_count,
        points[0].false_accept_count,
        points[0].false_nonmatch_count,
        points[0].true_nonmatch_count,
    ) == (0, 0, sample_count, sample_count)
    assert points[1].threshold == genuine[-1]
    assert (points[1].true_accept_count, points[1].false_accept_count) == (1, 0)
    assert points[-1].threshold == impostors[0]
    assert (
        points[-1].true_accept_count,
        points[-1].false_accept_count,
        points[-1].false_nonmatch_count,
        points[-1].true_nonmatch_count,
    ) == (sample_count, sample_count, 0, 0)
    assert points[-1].tar == 1.0
    assert points[-1].far == 1.0
    assert trapezoidal_auc(points) == pytest.approx(
        (sample_count + 1) / (2 * sample_count)
    )


def test_empty_scored_classes_and_nonfinite_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="no scored impostors"):
        calibrate_common_threshold({"sd300b": [None]}, 0.01)
    with pytest.raises(ValueError, match="scored genuine"):
        roc_det_points([None], [0.0])
    with pytest.raises(ValueError, match="finite"):
        compute_operating_metrics([math.nan], [0.0], 1.0)
