# SIFT Geometric Baseline

The repository exposes one SIFT method:

```text
method: sift_geometric
method_version: sift-geometric-v1
score_direction: higher_is_more_similar
```

It uses OpenCV's professional `cv2.SIFT_create` implementation. The Gaussian
pyramid, Difference of Gaussians, orientation assignment, and descriptor core
are not reimplemented in this repository. The method performs no decision or
thresholding in `compare()`; it returns one raw score and diagnostics. Frozen
development-only thresholds are applied later by reporting and cohort code.

## Historical reference findings

The previous repository was read only as an internal development reference.
Its relevant OpenCV pipeline was reconstructed exactly for the parity gate:

- grayscale loading with `cv2.IMREAD_GRAYSCALE`;
- CLAHE with clip limit 2.0 and an 8 by 8 tile grid;
- aspect-preserving resize to a 768-pixel long edge, centered in a black
  768 by 768 square with `INTER_AREA`, followed by min/max normalization;
- no Gaussian blur;
- `SIFT_create(nfeatures=3000, nOctaveLayers=3,
  contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)`;
- standard float32 SIFT descriptors;
- one-way `BFMatcher(NORM_L2, crossCheck=False)` KNN matching with `k=2`;
- strict Lowe test `first.distance < 0.75 * second.distance`;
- an early zero score when either descriptor array contained fewer than eight
  rows, even though affine estimation itself needs only three matches;
- at least three ratio matches for full affine estimation;
- `estimateAffine2D` with RANSAC threshold 3 pixels, maximum 2,000
  iterations, confidence 0.99, default refinement of 10 iterations, and an
  OpenCV RNG seed of zero;
- zero score for no descriptors, fewer than three matches, failed geometry,
  or zero inliers;
- raw score

  ```text
  inlier_count * (inlier_count / submitted_match_count) * log1p(submitted_match_count)
  ```

The old implementation treated missing descriptors as score zero. The new
benchmark contract instead records image or descriptor preparation problems as
explicit failures without a score. Insufficient matches or failed affine
verification remain a valid zero score with an explicit geometry failure
reason, because zero is part of the frozen score semantics rather than a
substitute for a preparation error.

Historical experiments showed that the composite score substantially improved
low-FAR TAR over keypoint-normalized inlier scoring on their selected
plain-to-roll protocol. They also exposed limitations that are addressed by
the new study design:

- cached image representations made old timing numbers unsuitable as
  cold-pair measurements;
- a fixed pixel RANSAC threshold was not comparable between 1000 and 2000 PPI;
- resize and CLAHE were changed together with matching and scoring, obscuring
  individual effects;
- exploratory score sweeps on the same selected pairs were not independent
  evaluation evidence;
- crop-grid probes could rescue genuine pairs but also introduced false
  accepts and therefore required validation-only guardrails;
- full affine, partial affine, RootSIFT, bidirectional matching, mutual
  consistency, and masking must be ablated one declared factor at a time.

Historical measurements are not inputs to selection on the new evaluation
subjects.

## New image and coordinate policy

The native baseline loads grayscale and does not resize, binarize, apply
CLAHE, blur, or write a processed image. Original dimensions and manifest PPI
are retained. Geometry maps coordinates to a 1000-PPI reference system before
RANSAC, so the configured reprojection threshold has the same physical meaning
for SD300b and SD300c.

The optional `valid_region` ablation is non-learned. It thresholds only
near-black padding/frame pixels, closes the support, keeps the largest
connected component, and applies PPI-scaled erosion. Coverage and keypoint
counts before and after the mask are recorded. It is disabled unless the
development ablation selects it.

RootSIFT is also optional and deterministic: L1 normalization, element-wise
square root, and an explicit all-zero result for a zero-norm descriptor.

## Matching and geometry diagnostics

The closed matching policies are one-way Lowe matching, bidirectional union,
and mutual Lowe consistency. Every comparison records forward and reverse KNN
counts, forward and reverse ratio counts, mutual count, and the exact count
submitted to geometry.

The closed geometry models are full and partial affine. Diagnostics include
inliers, outliers, inlier ratio, transform matrix, determinant, scale,
rotation, translation, reference-coordinate and destination-pixel residual
summaries, RANSAC parameters, and explicit failure reason. OpenCV does not
expose the realized RANSAC iteration count, so that field is recorded as null.

## Leakage controls and artifacts

Subjects are assigned by SHA-256 to one shared deterministic development or
evaluation partition across both datasets and all protocols. The shared pilot
contains genuine self, genuine plain-to-roll, and different-anatomical-identity
impostor pairs from development subjects only. Pipeline components, raw score,
and decision thresholds are frozen from this pilot before primary execution.

The six primary runs use the original manifests, serial cold-pair timing, two
prepares for every pair including self pairs, and no representation cache.
Bundles use the standard path:

```text
results/<dataset>/<protocol>/sift_geometric/pairwise-benchmark-v2/<config_hash>/
```

The SIFT-specific cohort is
`sift_geometric_joint_self_accept_v1`. Membership requires frozen acceptance in
all four self conditions and presence in both plain-to-roll manifests.
Plain-to-roll outcomes never affect membership and rejected pairs remain in
the filtered reports.
