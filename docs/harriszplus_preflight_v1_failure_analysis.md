# HarrisZ+ preflight v1 failure analysis

## Status

`harriszplus-rootsift-geometric-v1` remains a failed candidate. Its immutable
failure report is:

`results/harriszplus_rootsift_geometric/preflight/engineering_preflight_failure.json`

The report SHA-256 is
`9b822a9a2bc0e67e8b0bf3d9658b55d84865bb6c64f3f82a5900debebbb8cd42`.
No 500-identity pilot result was observed or published for v1.

## Frozen failure evidence

The failing case was `synthetic:noisy_corner`.

| Evidence | CPU | CUDA |
|---|---:|---:|
| Candidates after eigen-ratio filtering | 516 | 514 |
| Scale-1 candidates after eigen-ratio filtering | 13 | 11 |
| Final keypoints | 503 | 503 |

The scale-1 absolute candidate-count difference was 2. The legacy
minimum-to-maximum ratio was `0.846154`. The response-map gates passed, the
nearest-keypoint agreement was `0.996024`, and the matched response-rank
Spearman correlation was `0.999995916`. The CUDA repeat was exact.

## Cause

The v1 gate used only a relative count ratio. A ratio-only rule is unstable
when the denominator is small: an absolute difference of two candidates turns
`13` versus `11` into a large relative discrepancy even though the final
semantic output remained equivalent. The failure was therefore caused by the
validation contract, not by evidence of a changed algorithm or a changed
threshold decision.

## v2 disposition

The v2 candidate keeps every algorithmic parameter and the threshold-4
decision rule unchanged. Only the CPU/CUDA validation contract changes. It
uses a pre-declared absolute/hybrid count rule, final-keypoint semantic
equivalence, downstream CPU/CUDA decision equivalence, and exact CUDA-repeat
hashes. The v2 contract was frozen before executing its preflight and before
observing any HarrisZ+ 500-identity result.
