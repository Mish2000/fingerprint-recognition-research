# SourceAFIS Reproducibility Audit v1

The audit is an isolated technical reproducibility check. It does not calibrate
a match threshold, estimate recognition accuracy, relabel data, or replace the
full unfiltered benchmark result.

The workflow has three explicit phases:

1. `prepare` reads and validates the six existing benchmark-v2 bundles and
   freezes deterministic paired audit manifests. It never starts SourceAFIS.
2. `run` is the only phase that starts the managed Java sidecar. It writes six
   new bundles below `results/reproducibility_audits/` and cannot overwrite the
   primary bundles.
3. `compare` validates the rerun bundles and compares selected primary and
   rerun rows. It never starts SourceAFIS.

## Pre-specified strata

Selection is joint by `(subject_id, canonical_finger_position)` across SD300b
and SD300c for each protocol. The same identities are rerun at both
resolutions.

For every protocol, the audit includes:

- every identity with a zero score at both resolutions;
- every identity with a zero score only at SD300b;
- every identity with a zero score only at SD300c;
- the requested number of lowest positive identities among identities
  positive at both resolutions;
- a disjoint deterministic positive sample ranked by SHA-256 over the frozen
  seed, protocol, subject, and canonical finger position.

The default low-positive and deterministic-positive counts are each 100 per
protocol. All zero/discordant cases are included, regardless of count.

## Pass/fail policy

By default, reproducibility requires:

- exact numeric `raw_score` equality (`--score-abs-tolerance 0`);
- equal status and error code;
- equal prepare/compare diagnostics;
- equal SourceAFIS configuration hash;
- equal deterministic implementation hash, unless the explicit
  `--allow-jar-hash-variation` policy is selected;
- equal selected score-payload SHA-256 when the tolerance is zero.

Raw-score text equality is reported separately. Timings from both runs are
reported, but timing equality is never used for pass/fail.
When a positive tolerance is explicitly requested, the payload hash is still
reported but the numeric tolerance governs score pass/fail.

### Recorded JAR-hash variation in the primary bundles

The primary SD300b bundles record sidecar JAR SHA-256
`38952cfd0704b192cd0aebaf8606e6ad1ff1cca58c10a8cbd4f51e69ea72e9d2`.
The primary SD300c bundles and the currently retained sidecar JAR record
`84df3b736a6e7de1c4493e126433a7ac6aa92c174c24446ecb220ddf71a2712e`.
Every other deterministic implementation-hash component is identical.

The optional compatibility policy therefore permits only this field to vary.
It does not permit a different runner, contract, adapter, client, lifecycle,
declared SourceAFIS version, Maven coordinates, score semantics, or any other
implementation component. Both JAR hashes and the exact-hash mismatch remain
visible in the final report. Omit the flag to require byte-identical JAR
provenance.

## Safety and provenance

`prepare` records SHA-256 for every primary result, primary run metadata file,
source manifest, frozen audit manifest, and the selection table. `run` refuses
to start pair execution when these frozen hashes changed. Before each subset
run, the complete original manifest is validated against the read-only dataset,
then every subset row must equal its source-manifest row exactly.

Frozen plan artifacts are immutable: a repeated `prepare` succeeds only when
the bytes are identical. Completed run bundles use the benchmark-v2 candidate,
validation, and atomic-publication path. `--skip-existing` validates and reuses
completed isolated bundles, which supports safe recovery after interruption.

## Commands

Run from the repository root in the Python 3.11 conda environment:

```powershell
python -m fingerprint_benchmark.reproducibility_audit prepare `
  --primary-results-root results `
  --audit-root results\reproducibility_audits\sourceafis_reproducibility_audit_v1 `
  --seed sourceafis_reproducibility_audit_v1 `
  --low-positive-count 100 `
  --positive-sample-count 100

python -m fingerprint_benchmark.reproducibility_audit run `
  --audit-root results\reproducibility_audits\sourceafis_reproducibility_audit_v1 `
  --data-root C:\fingerprint-datasets `
  --sidecar-jar apps\sourceafis-sidecar\target\sourceafis-sidecar-0.2.0.jar `
  --skip-existing `
  --allow-jar-hash-variation

python -m fingerprint_benchmark.reproducibility_audit compare `
  --audit-root results\reproducibility_audits\sourceafis_reproducibility_audit_v1 `
  --score-abs-tolerance 0 `
  --allow-jar-hash-variation
```

For a guarded end-to-end invocation that stops immediately on any failed
phase, use:

```powershell
& .\scripts\run_sourceafis_reproducibility_audit.ps1
```

The comparison phase writes:

```text
results/reproducibility_audits/sourceafis_reproducibility_audit_v1/comparison/
  pair_comparison.csv
  condition_summary.csv
  audit_report.json
```
