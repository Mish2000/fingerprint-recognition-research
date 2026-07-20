# Full GFTT-Harris--RootSIFT geometric matcher v1

Status: implemented and parity-validated

## Identity and research scope

- `method_name`: `gftt_harris_rootsift_geometric`
- `method_version`: `gftt-harris-rootsift-geometric-v1`
- `score_direction`: `higher_is_more_similar`
- family: `handcrafted local-feature and geometric matching`
- parent pipeline: the Harris branch of `detector_only_v1`, frozen from the
  `detector_only_joint_500_v1` Joint-500 run metadata

This is a full-system local-feature fingerprint matcher using a single-scale
GFTT-Harris detector and RootSIFT descriptors.

The full method has its own identity, immutable configuration and provenance.
It deliberately reuses the protected detector-only implementation components
so that the established Harris result is reproduced exactly; it is not a new
detector-only branch and does not expose detector results as caller input.

## Source-of-truth component map

| Component | Current file | Current function/class | Configuration source | Full-system use |
| --- | --- | --- | --- | --- |
| grayscale loading | `src/fingerprint_benchmark/local_features/detector_only.py` | `DetectorOnlyAdapter.prepare` | OpenCV `IMREAD_GRAYSCALE` | reused through a private composed adapter |
| GFTT-Harris | `src/fingerprint_benchmark/detectors/opencv_gftt_harris.py` | `OpenCVGFTTHarrisDetector` | `OpenCVHarrisConfig` plus Joint-500 run metadata | reused; explicit full config is converted to detector config |
| point limiting and support | `src/fingerprint_benchmark/local_features/detector_only.py`, `support.py` | `canonical_features`, `assign_support_sizes` | `DetectorOnlyProtocolConfig` plus Joint-500 run metadata | reused; explicit full config is converted to common pipeline config |
| orientation | `src/fingerprint_benchmark/local_features/orientation.py` | `assign_orientations` | frozen `common_dominant_gradient_v1` policy | reused |
| supplied-keypoint SIFT | `src/fingerprint_benchmark/local_features/descriptors/sift_descriptor.py` | `compute_sift_descriptors` | frozen descriptor mode | reused |
| RootSIFT | `src/fingerprint_benchmark/local_features/descriptors/rootsift.py` | `rootsift` | frozen descriptor mode | reused |
| matching | `src/fingerprint_benchmark/local_features/matching.py` | `match_descriptors` | frozen Lowe/mutual fields | reused |
| geometry | `src/fingerprint_benchmark/local_features/geometry.py` | `verify_geometry` | frozen affine/RANSAC/PPI fields | reused |
| scoring | `src/fingerprint_benchmark/local_features/scoring.py` | `score_components`, `raw_score` | frozen inlier-count mode | reused |
| benchmark failure contract | `src/fingerprint_benchmark/local_features/detector_only.py`, `contract.py`, `runner.py` | `DetectorOnlyAdapter.prepare`, `compare`, `_execute_pair` | `pairwise-benchmark-v2` | wrapped without changing codes or outcome semantics |
| detector-study CLI and historical oracle | `src/fingerprint_benchmark/detector_only_joint500.py`, `cli.py` | `run_joint500`; `detector-joint500` | immutable Joint-500 manifests/results | not registered as the full method; used only by parity |
| full configuration and adapter | `src/fingerprint_benchmark/gftt_harris_full/config.py`, `adapter.py` | `GFTTHarrisRootSIFTGeometricConfig`, `GFTTHarrisRootSIFTGeometricAdapter` | explicit full-system v1 config | new independent identity and interface |
| full parity | `src/fingerprint_benchmark/gftt_harris_full/parity.py` | `run_parity` | immutable Joint-500 manifests and detector-only results | new exact-equivalence gate |

Authority order was current implementation, the frozen Joint-500 run config
and metadata, tests, then documentation. The run metadata and code agree.

## Input and preparation

`prepare(image_path, image_metadata)` reads the complete image with
`cv2.imread(path, cv2.IMREAD_GRAYSCALE)`. A readable image therefore enters the
algorithm as a non-empty, two-dimensional `uint8` grayscale array. No mask is
supplied. PPI is read from the manifest/image metadata and must be finite and
positive. The frozen study contains 1000 and 2000 PPI images; both retain their
native pixels. PPI changes physical support and coordinate normalization, not
the source-image resolution. The benchmark uses cold pairs with no
representation cache.

The full adapter accepts complete image A/image B paths and their respective
PPI metadata through the benchmark `prepare`/`compare` contract. A caller does
not supply points, orientations, descriptors, matches or a transform.

## Frozen GFTT-Harris detector

`cv2.goodFeaturesToTrack` is called with:

| OpenCV argument | Frozen value |
| --- | ---: |
| `maxCorners` | 3000 |
| `qualityLevel` | 0.01 |
| `minDistance` | 5.0 pixels |
| `blockSize` | 3 |
| `gradientSize` | 3 |
| `useHarrisDetector` | `True` |
| `k` | 0.04 |
| `mask` | `None` |

OpenCV owns corner border eligibility, response ranking and tie handling. The
implementation preserves `goodFeaturesToTrack` return order; no secondary sort
or tie-break is added. At most 3000 returned points are retained. The response
is unavailable through this API and is recorded as a zero diagnostic sentinel;
it never affects the representation.

## Single-scale support policy

Every retained point receives exactly one keypoint size:

```text
KeyPoint.size = 16.0 * image_ppi / 1000.0 pixels
```

Thus the native diameter is 16 pixels at 1000 PPI and 32 pixels at 2000 PPI,
representing the same physical diameter of 0.016 inch (0.4064 mm). There is no
octave search or multi-scale scale selection. This method is single-scale and
must not be described as scale-invariant.

## Orientation

The frozen policy is `common_dominant_gradient_v1`. OpenCV 3x3 Sobel gradients
are computed on the grayscale float32 image using `BORDER_REFLECT`. Each point
uses one 36-bin histogram. With `sigma = KeyPoint.size / 2`, the circular
support radius is `max(1, floor(3 * 1.5 * sigma + 0.5))`; samples are weighted
by a Gaussian whose sigma is `1.5 * sigma` and by gradient magnitude. The first
maximum bin selected by `numpy.argmax` defines an angle in degrees in
`[0, 360)`, using `atan2(gradient_y, gradient_x)`. Empty or zero-gradient
support falls back to 0 degrees. PPI affects orientation only through the
support size and radius.

## Descriptor and RootSIFT

The implementation constructs supplied `cv2.KeyPoint` objects from the Harris
locations, fixed support and computed orientations, then invokes
`cv2.SIFT.compute`; the SIFT detector is not invoked. The descriptor is 128
dimensional and returned/stored as contiguous float32. Its spatial support is
the supplied keypoint size described above.

For each validated SIFT descriptor `d`, RootSIFT is exactly:

```text
d_l1 = d / sum(abs(d))
root_d = sqrt(d_l1)
```

The sum is accumulated in float64, division output and the square-root result
are float32, and a zero L1 norm produces an all-zero descriptor.

## Matching

Matching uses `cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)` and KNN with
`k=2`. A forward neighbor is accepted only when
`first.distance < 0.75 * second.distance`. The same test is run in reverse and
only exact `(index_a, index_b)` pairs present in both directions are submitted
(`mutual`). Forward order is retained, so duplicates are not synthesized or
reordered. Preparation requires at least two descriptors per image. Geometry
requires at least three submitted matches.

## Geometry

The model is partial affine, estimated by `cv2.estimateAffinePartial2D` with
OpenCV RANSAC. Coordinates from each image are first multiplied by
`1000 / image_ppi`, so the 3.0-pixel reprojection threshold is expressed at the
1000-PPI reference. Frozen values are `maxIters=2000`, `confidence=0.99`,
`refineIters=10`, and OpenCV RNG seed 0 before estimation. The runtime is fixed
to 16 OpenCV threads with optimized code enabled. At least three
correspondences are required.

Geometry failure is a valid comparison: the transform is `None`, the inlier
mask is all false, and the failure reason is one of
`insufficient_geometry_matches`, `opencv_geometry_error`,
`model_estimation_failure`, or `invalid_inlier_mask`. A successful transform is
serialized as a 2x3 float matrix together with deterministic transform and
residual diagnostics.

## Raw score and decisions

The exact raw score is:

```text
raw_score = geometric_inlier_count
```

It is finite, non-negative and higher-is-more-similar. Geometry failure and a
successful model with no inliers both produce a valid score of zero. No
accept/reject threshold is applied by `compare()`. Any historical detector
study threshold belongs only to external reporting and is not part of this
algorithm.

## Failure semantics

Unreadable images are technical `image_read_failure` outcomes. Invalid dtype,
shape or PPI, detector/SIFT preparation errors and invalid supplied content use
`detector_only_preparation_failure`; fewer than two descriptors uses
`too_few_descriptors`. Invalid representation identity/version/format and
matching/geometry exceptions remain technical failures with their explicit
compatibility error codes. Technical failures have no raw score. Too few
matches or ordinary model-estimation failure is not technical failure: it is a
valid zero-score comparison with a geometry failure reason. Compatibility code
names are preserved deliberately because exact detector-only equivalence is a
release gate.

## Configuration identity and drift protection

The full-system dataclass states every algorithmic value explicitly and is
frozen. It is converted field-by-field to `OpenCVHarrisConfig` and
`DetectorOnlyProtocolConfig`; no downstream constructor default supplies an
algorithmic value. Its canonical algorithm representation excludes descriptive
metadata and is hashed as sorted compact JSON with SHA-256. Re-creating the
same config gives the same hash; changing any algorithmic field changes it.

An equivalence assertion compares the explicit conversion with the historical
Joint-500 Harris values. This protects the full v1 method from a later silent
change in detector-only defaults while still allowing both identities to reuse
the same tested pure components.

## Parity methodology

The parity gate reads the eight immutable `detector_only_joint_500_v1`
manifests (SD300b/SD300c crossed with plain-self, roll-self, plain-roll genuine
and plain-roll impostor). It selects at least ten rows from each using
`index_i = floor(i * N / 10)`, then adds available outcome/edge classes from the
historical detector-only result bundles. Each selected pair is executed through
both the existing detector-only Harris adapter and the new full adapter.

Status, score, error code, detector/descriptor/matching/geometry counts,
geometry failure reason, transform, representation hashes and every other
deterministic diagnostic are compared with exact equality. Timing, temporary
paths, process identity, method identity and implementation hash are excluded.
No numeric tolerance is used. The local report is written to
`results/restoration_preflight/gftt_harris_rootsift_geometric_v1/parity_report.json`
and is intentionally ignored by Git.

The implementation gate selected 93 pairs: 11 SD300b plain-self, 11 SD300b
roll-self, 12 each for the two SD300b plain-roll groups, 12 SD300c plain-self,
11 SD300c roll-self, and 12 each for the two SD300c plain-roll groups. All
93 current detector-only executions reproduced their historical oracle, and
all 93 full-system executions reproduced the current detector-only outcome and
deterministic diagnostics exactly. There were zero mismatches and no numeric
tolerance. The full canonical algorithm config SHA-256 is
`5765065c1f3f5238b47fa43746c3f6b2a2b271c6764834bdb9d28f0bcb8fd282`.

The untracked parity report is
`results/restoration_preflight/gftt_harris_rootsift_geometric_v1/parity_report.json`.
Its file SHA-256 is
`2dbf557d5a0132c994628c76742830a44eeb4728973362eb0f97c69a6d167306`;
the canonical report-object hash recorded inside it is
`e0a949fb09d2d49c38d482586d3fb8fee977790254bac3ade7032cf365f3959e`.

Cross-process repeatability used six distinct cases: positive plain-self,
positive roll-self, positive plain-roll genuine, zero-score plain-roll genuine,
plain-roll impostor, and a geometry failure. Every case was reconstructed in
three fresh processes. All 18 comparisons had identical status, raw score,
error code, deterministic diagnostics and representation hashes. The untracked
repeatability report file SHA-256 is
`8da0c4ad4b5cda1fd31f0d6dd2721bba6f26230db7fc39d17ad6b057b324cb7f`.

## Relationship to other methods and limitations

Unlike `sift_geometric-v1`, this method detects single-scale GFTT-Harris
locations and computes SIFT only at those supplied points; SIFT geometric uses
native multi-scale SIFT detection. Unlike the detector-only study, this is a
named full-system adapter with a method-owned immutable configuration and
provenance, not one detector branch in a shared ablation.

This is not a pure/original Harris matcher, a minutiae matcher, a
state-of-the-art claim or a final biometric-accuracy result. No threshold has
yet been selected. Accuracy evaluation, calibration and a shared full-system
development protocol are separate future work.
