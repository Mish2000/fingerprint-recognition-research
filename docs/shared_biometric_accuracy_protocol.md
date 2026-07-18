# Shared SourceAFIS/SIFT biometric accuracy protocol

This study is an accuracy/calibration protocol. It does not replace or alter
the existing `cold_pair` latency benchmark, and prepared-mode wall time must
not be interpreted as production latency.

## Primary comparison

Both methods are evaluated on the same held-out subjects, the same base
`plain_roll` genuine identities, the same logical impostor identity relations,
and the same FAR targets. Each method receives one threshold learned from
development only; that threshold must satisfy the target in both SD300b and
SD300c. The primary outcome is TAR at target FAR 1%, with 0.1% secondary.
FAR 0.01% is not claimed.

The pre-existing deterministic SIFT split is reused by reference as
`shared_biometric_accuracy_split_v1`. Its lists are rebuilt from the documented
SHA-256 subject rule and must match byte-for-byte before use. SD300b and
SD300c contain 192 development and 696 evaluation subjects with identical
identity membership.

## Shared pairs

Genuine pairs come only from:

- `protocols/sd300b/plain_roll.csv`
- `protocols/sd300c/plain_roll.csv`

For every genuine plain identity within a split, ten same-finger roll
identities from different subjects are selected. Selection is deterministic:
the shared split version, split, both identities, and a fixed seed are hashed
with SHA-256. Directional hash ordering makes only one orientation of an
unordered identity relation eligible, so reciprocal duplicates cannot occur.
The logical relation is generated once and then materialized independently to
the B and C image paths.

This yields, per dataset and method:

| Split | Genuine | Impostor |
|---|---:|---:|
| Development | 1,897 | 18,970 |
| Evaluation | 6,882 | 68,820 |

## Prepared-representation accuracy mode

The frozen adapters and raw-score definitions are unchanged. A separate
runner-owned cache prepares each unique image exactly once in a
method/dataset/split invocation. The cache is never shared between methods,
datasets, splits, or invocations. SIFT is sharded by canonical finger to bound
memory; the representation shard is deleted after all same-finger edges are
compared. No representation cache is published.

Score production performs no thresholding. Each score row records method,
config and implementation provenance, raw score/status/error, deterministic
diagnostics, image keys, cache scope, and genuine-score origin.

## Genuine reuse and provenance

All relevant primary bundles are fully validated. A mandatory preflight
recomputes 100 deterministic genuine pairs for every method/dataset and
requires exact score/status/diagnostic equality.

SIFT primary results use one uniform frozen implementation. The historical
SourceAFIS SD300b primary bundle used a different JAR from SD300c. Therefore
the shared study freezes the currently retained official SourceAFIS 3.18.1 JAR
for B and C, reuses only provenance-compatible genuine results, and recomputes
the SD300b genuine rows into the new namespace where strict same-JAR provenance
requires it. Existing bundles are never overwritten.

## Calibration, evaluation, and uncertainty

All scores are higher-is-more-similar and acceptance is inclusive:
`raw_score >= threshold`. For each target, the lowest common threshold whose
development FAR is at or below target in both datasets is frozen. SIFT uses
integer decision boundaries and accepts all ties at the boundary. Evaluation
scores cannot be produced until `frozen_thresholds.json` and the evaluation
safety gate exist.

Failures are reported separately and excluded from matcher-conditional rate
denominators. TAR, FNMR, and FAR include Wilson 95% intervals. Zero false
accepts additionally report the exact one-sided 95% upper bound rather than
claiming that the true FAR is zero. ROC/DET points are empirical and
unsmoothed; AUC is trapezoidal and EER is the nearest discrete operating point.

SD300b and SD300c are paired resolution conditions. Resolution reporting uses
aligned logical pair IDs, agreement/discordance counts, paired score deltas,
Pearson/Spearman correlations, and per-finger acceptance rates. They are not
treated as independent samples.

Legacy operating points remain separate: SourceAFIS threshold 40 and the
frozen SIFT threshold 4 rule. Development-only OR/AND fusion is exploratory
and is never used to rank the individual methods.

## Execution order

```powershell
fingerprint-shared-accuracy protect-before
fingerprint-shared-accuracy prepare
fingerprint-shared-accuracy preflight
fingerprint-shared-accuracy score-development
fingerprint-shared-accuracy calibrate
fingerprint-shared-accuracy score-evaluation
fingerprint-shared-accuracy report
# Run the full pytest suite, then record its exact result.
fingerprint-shared-accuracy record-tests --test-command "python -m pytest" --passed-count N --failed-count 0
fingerprint-shared-accuracy finalize
```

All outputs are written below
`results/shared_accuracy/sourceafis_sift_v1/`. Protected datasets, base
manifests, existing primary/derived results, pilots, thresholds, cohorts, and
cold-pair timing artifacts are SHA-256 inventoried before and after the study.
