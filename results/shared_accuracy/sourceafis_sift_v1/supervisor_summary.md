# Shared SourceAFIS/SIFT biometric accuracy — supervisor summary

This benchmark is separate from the existing cold-pair latency benchmark. Its primary outcome is TAR at a common target FAR using the same subjects, genuine pairs, impostor identity relations, and development/evaluation split.

## Cohort and pair counts

| Split | Subjects | Genuine per dataset | Impostors per dataset |
|---|---:|---:|---:|
| development | 192 | 1897 | 18970 |
| evaluation | 696 | 6882 | 68820 |

## Primary scientific comparison

| Method | Target FAR | Threshold | B TAR | B FAR | C TAR | C FAR | B FNMR | C FNMR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sourceafis | 1.000% | 14.608641525076788 | 77.027% | 0.949% | 76.736% | 0.881% | 22.973% | 23.264% |
| sourceafis | 0.100% | 24.93003517749199 | 72.406% | 0.087% | 72.552% | 0.071% | 27.594% | 27.448% |
| sift_geometric | 1.000% | 5 | 38.265% | 1.189% | 36.879% | 1.177% | 61.735% | 63.121% |
| sift_geometric | 0.100% | 14 | 10.420% | 0.144% | 9.459% | 0.131% | 89.580% | 90.541% |

## Interpretation and safeguards

- Frozen split: `shared_biometric_accuracy_split_v1` referencing the existing deterministic SIFT split.
- Impostor policy: 10 same-finger, different-subject roll identities per plain identity.
- Thresholds were selected on development scores only and then frozen before evaluation scoring.
- B and C are paired resolution conditions, not independent samples.
- Legacy operating points are reported separately from shared-FAR calibrated points.
- FAR 0.01% is not claimed; the prespecified primary/secondary targets are 1% and 0.1%.
- Prepared-mode wall runtime is not a cold-pair or production latency measurement.
- Pytest: 358 passed, 0 failed.
- Protected inputs unchanged: True.

## Ranking validity

The protocol now supports a valid like-for-like comparison at the same target FAR. Any method ranking is conditional on the two paired NIST resolution conditions and the reported confidence intervals; per-method derived-cohort acceptance and unmatched legacy thresholds are not used for ranking.
