# HarrisZ+ / RootSIFT geometric v3 postmortem

Generated from the eight frozen 500-pair bundles only. No detector, orientation,
descriptor, matcher, RANSAC, threshold, cap, PPI policy, or config was executed or changed.

## 1. Where genuine recognition collapses

The collapse is already visible before geometry and continues after RANSAC. Mean mutual
matches fall from 6.032 in B to
4.212 in C; mean inliers fall from
2.148 to
1.472. Accepted genuine pairs remain
64/500 in B and 1/500 in C at the frozen threshold 4.

## 2. Genuine failure-stage breakdown

| Stage | B | C |
|---|---:|---:|
| insufficient_descriptors | 0 | 0 |
| no_ratio_matches | 0 | 0 |
| insufficient_mutual_matches | 105 | 137 |
| ransac_not_attempted | 0 | 0 |
| ransac_model_failure | 0 | 0 |
| valid_model_but_0_to_3_inliers | 331 | 362 |
| accepted_4_or_more_inliers | 64 | 1 |

Classification is descriptive and does not alter saved status or score. Detailed pair-level
classification is in `failure_stage_classification.csv`; all requested distributions and
summary statistics are in `stage_attrition.csv`.

## 3. Paired B/C result

The same 500 identities were aligned by `(subject_id, canonical_finger_position)`.
Accepted in both: 1; accepted only B:
63; accepted only C:
0; rejected in both:
436. Score Pearson correlation is
0.175; median C-minus-B score delta is
0.000.

## 4. Physical-scale audit

At the same scale index, every pixel-defined support in C covers half the physical size of
B because B is 1000 PPI and C is 2000 PPI. This includes HarrisZ+ Gaussian support,
orientation radius, supplied-keypoint SIFT descriptor support, suppression distance, and
scale-0/1 duplicate removal. The full 10-row scale table is in
`physical_scale_audit.csv`.

The geometric layer is correctly PPI-normalized in all 4000 frozen
pair diagnostics: coordinates are converted to 1000-PPI reference pixels and the reference
threshold is 3 px. That is 3 native px in B and 6 native px in C, both exactly
0.0762 mm. This rules against a simple unnormalized RANSAC
threshold explanation.

## 5. Cap saturation

Across the 1,000 paired PLAIN/ROLL captures, the median pre-cap/final saturation factor is
8.999 in B and
32.414 in C; median C/B is
3.462. C is greater on
1000 captures, lower on 0, with
0 ties (two-sided exact sign-test p =
1.87e-301). Final B/C scale-mixture total-variation
distance is 0.068.

Candidate responses at ranks 2,990, 3,000 and 3,010 were not serialized, so their values
cannot be recovered without an impermissible rerun. The rank-window availability count and
the pre-cap scale proxy (`candidates_after_eigen_ratio`) are reported explicitly in
`cap_saturation.csv`.

## 6. Orientation diagnostics

All 2,000 self pairs have identical A/B representation hashes. The representation hash
includes the angle array, so this proves orientation-bearing payload identity on self
comparisons. Individual angles, histograms, peak dominance and ambiguity values were not
serialized; therefore entropy, 10-degree concentration, 0/180 concentration, B/C shift and
dispersion cannot be claimed. See `orientation_audit.json`.

## 7. Score histograms and descriptive threshold sensitivity

Frozen-threshold counts are B genuine 64/500, B
negative 3/500, C genuine
1/500, and C negative
0/500. `score_histograms.csv` reports bins 0, 1, 2, 3,
4, 5-9 and 10+, plus descriptive accepted counts at thresholds 1-4 for both genuine and
negative classes. `raw_scores.csv` preserves one row per frozen score. These calculations
do not select or change a threshold.

## 8. Existing SIFT backend comparison

The same 500 genuine identities per dataset were joined to the already-published SIFT
artifacts. In B, mean mutual matches are
6.032 for HarrisZ+ versus
16.646 for SIFT; mean inliers are
2.148 versus
7.378. In C, mean mutual matches are
4.212 versus
15.970; mean inliers are
1.472 versus
7.158. This is a diagnostic localization,
not a reranking.

## 9. Deterministic inspection

Selections are the first requested identities after sorting by subject and canonical finger.
Contact-sheet paths are enumerated in `inspection/selection.csv`. The sheets show B/C PLAIN
and ROLL source images, saved keypoint counts, mutual matches, RANSAC inputs, inliers,
scores and failure stages. Spatial overlays are not shown because coordinates were not
serialized.

## 10. Decision

**A. PPI/scale mismatch strongly supported.**

C loses genuine correspondences before RANSAC, fixed scale indices cover half the physical
support, the cap/scale behavior changes, and the RANSAC PPI normalization is verified.
This distinguishes implementation correctness (supported by preflight and deterministic
self behavior) from method suitability at 1000/2000 PPI (not supported by v3 results).

## 11. Exact next step

Create a new `v4 PPI-aware scale normalization` method/config. Select every parameter on
development identities outside these 500; keep v3 immutable; do not call v4 a small fix;
and treat any future 500-identity pilot as a demonstration report, not an independent
evaluation.

## 12. Integrity and provenance

Protected file count: 57. Protected tree SHA-256 before:
`2564be0521adf225adf18991087e10609af4313541237a623c16c32ce40ce31a`. After: `2564be0521adf225adf18991087e10609af4313541237a623c16c32ce40ce31a`. Result:
**byte-identical**. Generated artifact and source hashes are in `artifact_manifest.json`.

## 13. Explicit non-actions

No matcher or method was run. No HarrisZ+, SIFT, SourceAFIS, orientation or geometry was
rerun. No config, threshold, cap or PPI policy was changed. No commit or push was made by
this postmortem.
