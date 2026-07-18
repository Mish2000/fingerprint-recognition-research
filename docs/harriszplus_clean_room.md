# HarrisZ+ clean-room implementation statement

## Scope

The repository implementation of `harriszplus_rootsift_geometric` was written
from the mathematical description in the two papers listed below. No function,
class, or source fragment from the GPL reference implementation was copied into
this repository.

The external repository was used only as a behavioral oracle for conventions
that the papers leave ambiguous: Gaussian support and borders, sample standard
deviation, local-maximum handling, scale bookkeeping, sub-pixel behavior,
eigenvalue filtering, duplicate removal, and the intended uniform-selection
formula.

## Primary sources

- Fabio Bellavia and Dmytro Mishkin, *HarrisZ+: Harris Corner Selection for
  Next-Gen Image Matching Pipelines*, arXiv `2109.12925v6`, local PDF SHA-256
  `c9560b7e89849e7e4a05a4b376bc4eac2116eac8816b13fae9ff69859cb374df`.
- Fabio Bellavia et al., *Improving Harris corner selection strategy*, local
  `iet.pdf` SHA-256
  `513a8c18783ca7d6ce9b5beeecdfb796797df673a3289f303aa4124f079bcd86`.

## Reference oracle

- Repository: <https://github.com/fb82/HarrisZ>
- Inspected commit: `ec3a285ae3a86cc71d3026c01e24ec13554e3800`
- Commit date: 2025-07-18
- License: GNU GPL version 3
- Files inspected for behavior only: the repository's MATLAB HarrisZ/HarrisZ+
  implementation and the newer Python implementation.

The paper-era MATLAB implementation is treated as the primary oracle where the
two implementations differ. The Python implementation is secondary because it
contains known ranking, border, sub-pixel, and float-to-image-resize differences.

## Frozen clean-room conventions

- Input is native-resolution, single-channel grayscale in the source 0-255
  range and is processed as float32 throughout the detector. There is no
  enhancement, segmentation, binarization, downsampling, or hidden resize.
- Coordinates are zero-based OpenCV pixel centers in `(x, y)` order.
- Central differences implement mathematical convolution by `[-1, 0, 1]`; the
  outside one-pixel derivative frame is zero.
- Gaussian kernels are normalized sampled Gaussians with radius
  `max(1, ceil(3*sigma))`. Padding is symmetric and includes the edge sample,
  matching OpenCV `BORDER_REFLECT` and the MATLAB oracle.
- Z-score normalization uses the sample standard deviation (`N-1`). A map with
  zero standard deviation maps to zero. Each dense scale counts non-finite
  responses before any candidate selection and raises instead of sanitizing a
  non-finite result.
- Scale indexes are `0..4`, with integration sigma `2^(i/2)` and differentiation
  sigma `2^((i-1)/2)`. Indexes 0 and 1 run on a deterministic 2x image and are
  mapped back to the original coordinate system.

  | index | working image | working `(sigma_d, sigma_i)` | output `(sigma_d, sigma_i)` | suppression distance (working px) |
  |---:|---:|---:|---:|---:|
  | 0 | 2x | `(sqrt(2), 2)` | `(1, sqrt(2))` | 5 |
  | 1 | 2x | `(2, 2*sqrt(2))` | `(1, sqrt(2))` | 6 |
  | 2 | native | `(sqrt(2), 2)` | `(sqrt(2), 2)` | 5 |
  | 3 | native | `(2, 2*sqrt(2))` | `(2, 2*sqrt(2))` | 6 |
  | 4 | native | `(2*sqrt(2), 4)` | `(2*sqrt(2), 4)` | 9 |

- The 2x image is produced once with OpenCV `INTER_LANCZOS4`, as required by the
  project specification. This is a recorded convention difference from the
  oracle's three-lobe Lanczos resize.
- Index 0 receives the index-1 output scale. Indexes 0 and 1 therefore both
  output `(sigma_d, sigma_i) = (1, sqrt(2))`.
- Candidates require strict `H > 0` and strict smoothed edge mask `M > 0.31`.
  The raw edge indicator remains the paper's strict gradient-magnitude-greater-
  than-global-mean comparison. Equal maxima in the local 7x7 neighborhood are
  rejected. A frozen selection-only absolute tolerance of `1e-6` treats tiny
  float32 backend splits of one mathematical plateau as a tie; raw response
  maps and returned response values are never quantized.
- Per-scale greedy suppression uses the differentiation-scale support radius;
  a distance exactly equal to the radius survives.
- Sub-pixel refinement uses independent three-sample parabolas. There is no
  clipping. A border point or invalid/zero denominator receives zero offset.
- The eigen/axis filter keeps a point only when
  `sqrt(lambda_min/lambda_max) > 0.25`; non-positive or non-finite eigenvalues
  are rejected.
- Final ranking is response descending, scale index descending, `y` ascending,
  `x` ascending, and source candidate index ascending.
- Nearly duplicate index-0/index-1 points less than one source pixel apart are
  suppressed; distance exactly one survives.
- The printed Eq. 14 is dimensionally inconsistent. Its prose derivation and
  both official implementations establish the intended formula as
  `q = sqrt(8*m*n/(pi*k))`. Uniform selection applies this distance within each
  greedy pass, then re-runs on discarded points until the hard cap is filled.
- The hard cap is exactly 3,000 keypoints before descriptor extraction.
- The papers output an affine support region and do not specify a scalar OpenCV
  keypoint size. This adapter freezes the natural SIFT mapping
  `KeyPoint.size = 2*sigma_integration`; it is an adapter choice, not a claim
  about paper behavior. Every supplied native-coordinate OpenCV keypoint uses
  `octave=0`; `class_id` carries the HarrisZ+ scale index, while source index
  and full scale/support metadata are retained separately.

CPU/CUDA dense-response validation keeps the independently frozen numerical
tolerances (`atol=5e-4`, `rtol=2e-4`). Each scale must have minimum, maximum,
mean, and sample standard deviation within those tolerances, and at least
`0.9999` of real-image pixels must be individually all-close. Synthetic maps
require coverage `1.0`. Outlier count, fraction, and maximum delta are always
recorded, and a hard safety bound requires the maximum absolute delta to be no
more than `0.1` (the fixed real-image audit observed a worst value of about
`0.0888`). This statistical rule preserves the paper's discontinuous binary
edge test: on a few full-resolution pixels, backend rounding can place a
gradient on opposite sides of the exact global mean without changing the
selected keypoints. Every per-scale and aggregate candidate-stage count is
also compared and reported: pre-uniform minimum/maximum count ratios must be
at least `0.9995`, while uniform-selection and final counts must be exact.
Nearest-unused keypoint agreement must be at least `0.95`. Ordering is a
separate gate: after nearest correspondence, validation-only canonical ranks
may reorder only *contiguous* response groups whose full range is at most
`1e-6`, with coordinate ties at `1e-3` px; Spearman rank correlation between
the matched CPU/CUDA ranks must be at least `0.99`. Thus a reversed or random
ordering of distinct responses fails. Raw ordered-position agreement and a
canonical ordered-position projection are always reported as evidence, but are
not substituted for the correlation gate. This prevents symmetric checkerboard
corners, and small local rank jitter in the noisy-corner case, from failing only
because float32 refinement changes a mathematical tie; the raw detector list
and response values remain unchanged. These validation gates were finalized
during development validation, then fixed before the formal immutable
preflight and before any 500-identity result; they may not be relaxed afterward.
Timing and peak-memory telemetry remain evidence but are excluded from
deterministic repeat hashes. Peak allocated/reserved memory must be finite,
positive, internally consistent, and no more than `0.90` of device VRAM.
The frozen algorithm and runner configurations bind `arXiv:2109.12925v6`, the
float32 detector dtype, all strict threshold policies, and an explicit five-row
derived scale table. `detector_gpu_kernel_ms` is the sum of synchronized CUDA-
event elapsed times for base gradients, all dense scales, and GPU eigen/subpixel
refinement; transfer and CPU selection wall times are recorded separately.

## Pipeline boundary

Only the detector family changes. The implemented study intentionally does not
reproduce the paper's full matching pipeline:

- deterministic single-orientation SIFT-style assignment is added because the
  supplied OpenCV descriptor must not receive `angle=-1`;
- existing repository RootSIFT normalization is reused unchanged;
- existing mutual BF-L2 Lowe-ratio matching (`k=2`, ratio `0.75`) is reused
  unchanged instead of FGINN;
- existing PPI-normalized partial-affine RANSAC is reused unchanged;
- the raw score is the number of geometric inliers;
- decision thresholding is outside the adapter.

No AffNet, HardNet, DeDoDe, XFeat, AdaLAM, DTM, learned weights, or learned
model is used.

## CUDA runtime dependency

The optional project dependency is pinned as `torch==2.11.0`. The validated
Windows/Blackwell environment uses the official CUDA 13.0 wheel:

```powershell
python -m pip install torch==2.11.0 `
  --index-url https://download.pytorch.org/whl/cu130
```

No network access is needed by the detector or pilot after environment setup.
