# Detector-only joint-500 screening protocol

## Research question

`detector_only_joint_500_v1` asks whether final minutia locations selected by
SourceAFIS are better RootSIFT anchors than OpenCV GFTT-Harris locations when
every downstream operation is identical: common physical support,
`common_dominant_gradient_v1`, supplied-keypoint SIFT computation, RootSIFT,
mutual Lowe-ratio matching, PPI-normalized affine RANSAC, and geometric inlier
count.

This is separate from the SourceAFIS end-to-end matcher. The SourceAFIS-location
branch uses no SourceAFIS matching score, threshold, or decision.

## Method-neutral cohort selection

Selection reads only the six committed source manifests: plain self, roll self,
and genuine plain-roll for both SD300b and SD300c. It never reads results,
scores, detector output, templates, failure lists, or historical cohorts.

An identity is `(subject_id, canonical_finger_position)`. It is eligible only
when it occurs exactly once in all six source manifests and the plain, roll,
genuine, raw-FRGP, PPI, and path relationships are internally consistent.

The frozen seed is `detector-only-joint-500-v1`. For finger positions 1 through
10, candidates are ranked by SHA-256 over newline-separated protocol version,
seed, finger position, and subject ID, with subject ID as tie-breaker. The first
50 candidates whose subject has not been used for an earlier position are
selected. Original row order is not a selection policy.

The resulting cohort has exactly 500 identities, 50 per finger position, and
500 unique subjects. One finger is selected per subject. SD300b and SD300c use
the same logical identities.

## Genuine, impostor, and self pairs

Genuine rows are the exact selected source plain-roll rows with only derived
pair/protocol identity fields changed. Within each finger position, the 50
identities are ordered by selection rank and paired by circular shift one:
Plain from subject A is compared with Roll from the next subject B. This is a
full bijection, A never equals B, finger position is unchanged, and B/C use the
same logical pairing.

Plain-self and roll-self rows are engineering diagnostics only. A self failure
is recorded, but it never removes or replaces an identity and never changes a
genuine or impostor row.

## SourceAFIS final-template semantics

The detector consumes exact raw grayscale bytes and parses the documented
native template produced by SourceAFIS 3.18.1. The native minutiae are the final
selected set after deterministic shuffling. Template order is not a quality
ranking. Direction and ending/bifurcation type are retained only as diagnostics
and do not enter the common downstream representation.

Scaled SourceAFIS coordinates are mapped using actual template/native image
dimensions and pixel centers:

```text
x_native = ((x_scaled + 0.5) * native_width / scaled_width) - 0.5
y_native = ((y_scaled + 0.5) * native_height / scaled_height) - 0.5
```

There is no integer rounding, theoretical-DPI substitute, or silent clipping.

## Score and screening interpretation

Both detector branches use `detector_only_v1` unchanged and return raw
`geometric_inlier_count`. No adapter threshold exists. Raw score magnitude must
not be compared across methods, because detector point count is itself part of
detector output and SourceAFIS/Harris may return different counts.

The report is screening-only. Genuine and impostor rows produce an empirical
ROC, AUC, screening EER, and a report-only operating point at FAR 1%. With 500
impostors, FAR resolution is 0.002; FAR 0.1% is not reported or claimed.
Threshold calibration is `none`.

SD300b and SD300c are paired views joined by logical identity/pair, not pooled as
independent observations. This is a development/screening cohort, not held-out
evaluation. A negative result applies only to minutia locations under the common
RootSIFT pipeline; it is not a test of minutiae-native matching.
