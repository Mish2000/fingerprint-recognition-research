"""Pure metrics for the shared biometric accuracy protocol.

All decisions use the higher-is-more-similar rule ``score >= threshold``.
``None`` is the only failure marker accepted by this module.  Failures are
reported separately and are never silently converted into rejects or scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from numbers import Real
from statistics import NormalDist
from typing import Iterable, Mapping, TypeAlias


NumericScore: TypeAlias = int | float
ScoreObservation: TypeAlias = NumericScore | None


@dataclass(frozen=True)
class BinomialInterval:
    """A binomial proportion interval."""

    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class CalibrationDatasetMetrics:
    """Impostor outcome for one dataset at a calibrated common threshold."""

    total_count: int
    scored_count: int
    failure_count: int
    false_accept_count: int
    true_nonmatch_count: int
    scores_at_threshold: int
    far: float


@dataclass(frozen=True)
class CommonThresholdCalibration:
    """One threshold constrained by every supplied development dataset."""

    threshold: NumericScore
    target_far: float
    integer_scores: bool
    candidate_count: int
    per_dataset: Mapping[str, CalibrationDatasetMetrics]


@dataclass(frozen=True)
class OperatingMetrics:
    """Genuine and impostor outcomes at one frozen operating threshold."""

    threshold: NumericScore
    genuine_total_count: int
    genuine_scored_count: int
    genuine_failure_count: int
    impostor_total_count: int
    impostor_scored_count: int
    impostor_failure_count: int
    true_accept_count: int
    false_nonmatch_count: int
    false_accept_count: int
    true_nonmatch_count: int
    genuine_scores_at_threshold: int
    impostor_scores_at_threshold: int
    tar: float | None
    fnmr: float | None
    far: float | None
    tar_wilson_95: BinomialInterval | None
    fnmr_wilson_95: BinomialInterval | None
    far_wilson_95: BinomialInterval | None
    zero_false_accept_upper_95: float | None


@dataclass(frozen=True)
class RocDetPoint:
    """One empirical, unsmoothed ROC/DET operating point."""

    threshold: NumericScore
    true_accept_count: int
    false_nonmatch_count: int
    false_accept_count: int
    true_nonmatch_count: int
    genuine_scores_at_threshold: int
    impostor_scores_at_threshold: int
    genuine_failure_count: int
    impostor_failure_count: int
    tar: float
    fnmr: float
    far: float


@dataclass(frozen=True)
class DiscreteEerEstimate:
    """Closest empirical ROC/DET point to equal error rates."""

    threshold: NumericScore
    eer: float
    far: float
    fnmr: float
    absolute_gap: float


def calibrate_common_threshold(
    impostor_scores_by_dataset: Mapping[str, Iterable[ScoreObservation]],
    target_far: float,
    *,
    integer_scores: bool = False,
) -> CommonThresholdCalibration:
    """Choose the lowest common score boundary satisfying FAR in every dataset.

    Candidate boundaries are deterministic.  Continuous scores use the unique
    observed scores plus the next representable value above the global maximum.
    Integer scores use every integral boundary from the observed minimum through
    ``global_max + 1``; this includes unobserved gaps and makes the selected
    boundary the lowest valid integer.  The final candidate explicitly accepts
    none.  Ties are accepted together because the operator is ``>=``.

    ``None`` observations are counted as failures but excluded from empirical
    FAR denominators.  Every dataset must contain at least one scored impostor.
    """

    _validate_probability(target_far, name="target_far", upper_inclusive=False)
    if not impostor_scores_by_dataset:
        raise ValueError("At least one dataset is required for common calibration.")

    prepared: dict[str, tuple[tuple[NumericScore, ...], int]] = {}
    pooled: list[NumericScore] = []
    for dataset in sorted(impostor_scores_by_dataset):
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Dataset names must be non-empty strings.")
        scores, failure_count = _coerce_observations(
            impostor_scores_by_dataset[dataset], integer_scores=integer_scores
        )
        if not scores:
            raise ValueError(f"Dataset {dataset!r} has no scored impostors.")
        prepared[dataset] = (scores, failure_count)
        pooled.extend(scores)

    candidates = _threshold_candidates(pooled, integer_scores=integer_scores)
    threshold: NumericScore | None = None
    for candidate in candidates:
        if all(
            _accepted_count(scores, candidate) / len(scores) <= target_far
            for scores, _ in prepared.values()
        ):
            threshold = candidate
            break
    if threshold is None:  # The accept-none candidate makes this unreachable.
        raise AssertionError("No common threshold candidate satisfied the target FAR.")

    per_dataset: dict[str, CalibrationDatasetMetrics] = {}
    for dataset, (scores, failure_count) in prepared.items():
        false_accept_count = _accepted_count(scores, threshold)
        scored_count = len(scores)
        per_dataset[dataset] = CalibrationDatasetMetrics(
            total_count=scored_count + failure_count,
            scored_count=scored_count,
            failure_count=failure_count,
            false_accept_count=false_accept_count,
            true_nonmatch_count=scored_count - false_accept_count,
            scores_at_threshold=sum(score == threshold for score in scores),
            far=false_accept_count / scored_count,
        )

    return CommonThresholdCalibration(
        threshold=threshold,
        target_far=float(target_far),
        integer_scores=integer_scores,
        candidate_count=len(candidates),
        per_dataset=per_dataset,
    )


def compute_operating_metrics(
    genuine_scores: Iterable[ScoreObservation],
    impostor_scores: Iterable[ScoreObservation],
    threshold: NumericScore,
    *,
    integer_scores: bool = False,
) -> OperatingMetrics:
    """Compute TAR/FNMR/FAR while keeping comparison failures separate.

    Rates use scored comparisons as their denominators.  A ``None`` observation
    increments the relevant failure count and contributes to no decision count.
    """

    checked_threshold = _coerce_threshold(threshold, integer_scores=integer_scores)
    genuine, genuine_failures = _coerce_observations(
        genuine_scores, integer_scores=integer_scores
    )
    impostors, impostor_failures = _coerce_observations(
        impostor_scores, integer_scores=integer_scores
    )

    true_accepts = _accepted_count(genuine, checked_threshold)
    false_nonmatches = len(genuine) - true_accepts
    false_accepts = _accepted_count(impostors, checked_threshold)
    true_nonmatches = len(impostors) - false_accepts
    tar = _optional_rate(true_accepts, len(genuine))
    fnmr = _optional_rate(false_nonmatches, len(genuine))
    far = _optional_rate(false_accepts, len(impostors))

    return OperatingMetrics(
        threshold=checked_threshold,
        genuine_total_count=len(genuine) + genuine_failures,
        genuine_scored_count=len(genuine),
        genuine_failure_count=genuine_failures,
        impostor_total_count=len(impostors) + impostor_failures,
        impostor_scored_count=len(impostors),
        impostor_failure_count=impostor_failures,
        true_accept_count=true_accepts,
        false_nonmatch_count=false_nonmatches,
        false_accept_count=false_accepts,
        true_nonmatch_count=true_nonmatches,
        genuine_scores_at_threshold=sum(score == checked_threshold for score in genuine),
        impostor_scores_at_threshold=sum(score == checked_threshold for score in impostors),
        tar=tar,
        fnmr=fnmr,
        far=far,
        tar_wilson_95=_optional_wilson(true_accepts, len(genuine)),
        fnmr_wilson_95=_optional_wilson(false_nonmatches, len(genuine)),
        far_wilson_95=_optional_wilson(false_accepts, len(impostors)),
        zero_false_accept_upper_95=(
            exact_zero_false_accept_upper_bound(len(impostors))
            if impostors and false_accepts == 0
            else None
        ),
    )


def wilson_interval(
    success_count: int,
    trial_count: int,
    *,
    confidence: float = 0.95,
) -> BinomialInterval:
    """Return the two-sided Wilson score interval for a binomial rate."""

    _validate_binomial_counts(success_count, trial_count)
    _validate_probability(
        confidence,
        name="confidence",
        lower_exclusive=True,
        upper_inclusive=False,
    )
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    observed = success_count / trial_count
    denominator = 1.0 + z_squared / trial_count
    center = (observed + z_squared / (2.0 * trial_count)) / denominator
    radius = (
        z
        * math.sqrt(
            observed * (1.0 - observed) / trial_count
            + z_squared / (4.0 * trial_count * trial_count)
        )
        / denominator
    )
    return BinomialInterval(
        lower=max(0.0, center - radius),
        upper=min(1.0, center + radius),
        confidence=float(confidence),
    )


def exact_zero_false_accept_upper_bound(
    impostor_count: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Return the exact one-sided binomial upper bound for zero false accepts.

    This is the zero-success Clopper-Pearson bound
    ``1 - (1 - confidence) ** (1 / impostor_count)``.
    """

    if type(impostor_count) is not int or impostor_count <= 0:
        raise ValueError("impostor_count must be a positive integer.")
    _validate_probability(
        confidence,
        name="confidence",
        lower_exclusive=True,
        upper_inclusive=False,
    )
    return -math.expm1(math.log1p(-confidence) / impostor_count)


def roc_det_points(
    genuine_scores: Iterable[ScoreObservation],
    impostor_scores: Iterable[ScoreObservation],
    *,
    integer_scores: bool = False,
) -> tuple[RocDetPoint, ...]:
    """Build deterministic empirical ROC and DET points without smoothing.

    Points are ordered from the accept-none threshold toward accept-all.  DET
    coordinates are available directly as ``far`` and ``fnmr`` on each point.
    Failures remain separate and do not enter either denominator.
    """

    genuine, genuine_failures = _coerce_observations(
        genuine_scores, integer_scores=integer_scores
    )
    impostors, impostor_failures = _coerce_observations(
        impostor_scores, integer_scores=integer_scores
    )
    if not genuine:
        raise ValueError("At least one scored genuine comparison is required.")
    if not impostors:
        raise ValueError("At least one scored impostor comparison is required.")

    unique_scores = sorted(set((*genuine, *impostors)), reverse=True)
    maximum = unique_scores[0]
    accept_none: NumericScore
    if integer_scores:
        accept_none = int(maximum) + 1
    else:
        accept_none = math.nextafter(float(maximum), math.inf)
    thresholds = (accept_none, *unique_scores)

    genuine_counts = Counter(genuine)
    impostor_counts = Counter(impostors)
    true_accepts = 0
    false_accepts = 0
    points: list[RocDetPoint] = []
    for index, threshold in enumerate(thresholds):
        if index:
            true_accepts += genuine_counts[threshold]
            false_accepts += impostor_counts[threshold]
        false_nonmatches = len(genuine) - true_accepts
        true_nonmatches = len(impostors) - false_accepts
        points.append(
            RocDetPoint(
                threshold=threshold,
                true_accept_count=true_accepts,
                false_nonmatch_count=false_nonmatches,
                false_accept_count=false_accepts,
                true_nonmatch_count=true_nonmatches,
                genuine_scores_at_threshold=genuine_counts[threshold],
                impostor_scores_at_threshold=impostor_counts[threshold],
                genuine_failure_count=genuine_failures,
                impostor_failure_count=impostor_failures,
                tar=true_accepts / len(genuine),
                fnmr=false_nonmatches / len(genuine),
                far=false_accepts / len(impostors),
            )
        )
    return tuple(points)


def trapezoidal_auc(points: Iterable[RocDetPoint]) -> float:
    """Integrate TAR over FAR for ordered empirical ROC points."""

    ordered = tuple(points)
    if len(ordered) < 2:
        raise ValueError("At least two ROC points are required for AUC.")
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        if right.far < left.far or right.tar < left.tar:
            raise ValueError("ROC points must be monotone in FAR and TAR.")
        area += (right.far - left.far) * (left.tar + right.tar) / 2.0
    return min(1.0, max(0.0, area))


def discrete_eer(points: Iterable[RocDetPoint]) -> DiscreteEerEstimate:
    """Return the empirical point minimizing ``abs(FAR - FNMR)``.

    When multiple points have the same gap, the highest threshold wins.  The
    reported EER is the arithmetic mean of FAR and FNMR at that discrete point;
    no interpolation or smoothing is performed.
    """

    available = tuple(points)
    if not available:
        raise ValueError("At least one ROC/DET point is required for EER.")
    selected = min(
        available,
        key=lambda point: (abs(point.far - point.fnmr), -float(point.threshold)),
    )
    gap = abs(selected.far - selected.fnmr)
    return DiscreteEerEstimate(
        threshold=selected.threshold,
        eer=(selected.far + selected.fnmr) / 2.0,
        far=selected.far,
        fnmr=selected.fnmr,
        absolute_gap=gap,
    )


def _coerce_observations(
    observations: Iterable[ScoreObservation],
    *,
    integer_scores: bool,
) -> tuple[tuple[NumericScore, ...], int]:
    scores: list[NumericScore] = []
    failure_count = 0
    for index, observation in enumerate(observations):
        if observation is None:
            failure_count += 1
            continue
        try:
            scores.append(_coerce_score(observation, integer_scores=integer_scores))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Invalid score at index {index}: {exc}") from exc
    return tuple(scores), failure_count


def _coerce_score(score: NumericScore, *, integer_scores: bool) -> NumericScore:
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError("scores must be real numbers or None failures.")
    numeric = float(score)
    if not math.isfinite(numeric):
        raise ValueError("scores must be finite.")
    if integer_scores:
        if not numeric.is_integer():
            raise ValueError("integer_scores=True requires integral score values.")
        return int(numeric)
    return numeric


def _coerce_threshold(threshold: NumericScore, *, integer_scores: bool) -> NumericScore:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number.")
    numeric = float(threshold)
    if math.isnan(numeric) or numeric == -math.inf:
        raise ValueError("threshold must not be NaN or negative infinity.")
    if integer_scores:
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("An integer-score threshold must be a finite integer.")
        return int(numeric)
    return numeric


def _threshold_candidates(
    scores: Iterable[NumericScore],
    *,
    integer_scores: bool,
) -> tuple[NumericScore, ...]:
    unique = sorted(set(scores))
    if not unique:
        raise ValueError("At least one score is required to build thresholds.")
    if integer_scores:
        # Scores are integral and acceptance is inclusive.  A missing integer
        # between two observed scores is still a legitimate lower decision
        # boundary and must be considered when selecting the *lowest*
        # satisfying threshold.
        return tuple(range(int(unique[0]), int(unique[-1]) + 2))
    else:
        accept_none = math.nextafter(float(unique[-1]), math.inf)
    if accept_none == unique[-1]:
        raise ValueError("Could not construct an accept-none threshold.")
    return (*unique, accept_none)


def _accepted_count(scores: Iterable[NumericScore], threshold: NumericScore) -> int:
    return sum(score >= threshold for score in scores)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _optional_wilson(success_count: int, trial_count: int) -> BinomialInterval | None:
    return wilson_interval(success_count, trial_count) if trial_count else None


def _validate_binomial_counts(success_count: int, trial_count: int) -> None:
    if type(success_count) is not int or type(trial_count) is not int:
        raise ValueError("Binomial counts must be integers.")
    if trial_count <= 0:
        raise ValueError("trial_count must be a positive integer.")
    if success_count < 0 or success_count > trial_count:
        raise ValueError("success_count must be between zero and trial_count.")


def _validate_probability(
    value: float,
    *,
    name: str,
    lower_exclusive: bool = False,
    upper_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number.")
    lower_ok = value > 0.0 if lower_exclusive else value >= 0.0
    upper_ok = value <= 1.0 if upper_inclusive else value < 1.0
    if not lower_ok or not upper_ok:
        lower = "0 <" if lower_exclusive else "0 <="
        upper = "<= 1" if upper_inclusive else "< 1"
        raise ValueError(f"{name} must satisfy {lower} {name} {upper}.")


__all__ = [
    "BinomialInterval",
    "CalibrationDatasetMetrics",
    "CommonThresholdCalibration",
    "DiscreteEerEstimate",
    "NumericScore",
    "OperatingMetrics",
    "RocDetPoint",
    "ScoreObservation",
    "calibrate_common_threshold",
    "compute_operating_metrics",
    "discrete_eer",
    "exact_zero_false_accept_upper_bound",
    "roc_det_points",
    "trapezoidal_auc",
    "wilson_interval",
]
