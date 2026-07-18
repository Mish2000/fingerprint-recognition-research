# HarrisZ+ preflight v2 failure analysis

## Status

`harriszplus-rootsift-geometric-v2` remains a failed candidate. Its immutable
contract and failure report are:

- `results/harriszplus_rootsift_geometric_v2/preflight/engineering_preflight_contract_v2.json`
- `results/harriszplus_rootsift_geometric_v2/preflight/engineering_preflight_failure.json`

The contract SHA-256 is
`3dfa83653375763eedc9b6df40168d1d7b686d727613af54390f53c16540db9a`.
The failure-report SHA-256 is
`db18ba8747de4a436fcff78e259ca2d56aa0f639c4f9febe3bee0d2f52d953ce`.
No HarrisZ+ 500-identity result was observed or published for v2.

## Frozen failure evidence

The only failing gate was the final-response Spearman threshold on
`synthetic:noisy_corner`.

| Evidence | Result |
|---|---:|
| CPU/CUDA final keypoints | 503 / 503 |
| Bidirectional matched fraction | 1.0 |
| Maximum matched spatial delta | 0.123553 px |
| Maximum matched relative scale delta | 0 |
| Response-rank Spearman | 0.999700809 |
| Exact CPU/CUDA raw scores | 10 / 10 |

Statuses, failure stages, and threshold-4 decisions were identical for all ten
real comparisons. The CUDA detector, real representations, comparisons, raw
scores, decisions, statuses, and deterministic diagnostics repeated exactly.

## Cause

The v2 Spearman gate measured relative ordering among responses that were
nearly tied. Small CPU/CUDA floating-point differences changed the order of a
few matched keypoints even though the final keypoint sets, spatial/scale
payload, descriptors, downstream scores, and decisions were functionally
equivalent.

## v3 disposition

V3 retains global, per-scale, and top-response Spearman values as diagnostics
without a minimum pass threshold. Functional equivalence remains gated by
pre-declared response-map tolerances, candidate-count rules, bidirectional
keypoint overlap, top-K overlap when the cap is active, descriptor validity,
downstream score/decision equivalence, and exact CUDA reproducibility.

