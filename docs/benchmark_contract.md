# Pairwise Benchmark Contract v2

The explicit contract identifier is:

```text
pairwise-benchmark-v2
```

It is part of run configuration, the immutable run specification, every result
row, run metadata, and the output path. Historical SourceAFIS v1 files remain at
their original paths and are never replaced.

## Scope

This milestone produces only raw pairwise method scores, operation timings, and
explicit operation status. It does not add thresholds, negative pairs, TA/FR,
ROC/DET/EER, identification, fusion, SIFT, NBIS, Harris, TPS, MCC, deep methods,
or parallel execution.

## Output identity

Every primary bundle is published under:

```text
results/<dataset>/<protocol>/<method>/pairwise-benchmark-v2/<config_hash>/
```

The directory contains exactly the canonical `pairs.csv` and
`run_metadata.json` artifacts. A reproducibility rerun can use a separate
`--results-root`; it retains the same contract-relative layout without
overwriting the primary bundle.

`config_hash` is SHA-256 over canonical JSON containing method configuration,
method/version, score semantics, timing mode, warm-up policy, and contract
version. `implementation_hash` is a separate SHA-256 over stable
implementation facts. The run specification binds:

- expected dataset and protocol;
- resolved manifest path and current manifest SHA-256;
- method and method version;
- benchmark contract version;
- config hash;
- implementation hash.

## Method-neutral adapter

Representations are genuinely opaque. Their payload is `Any`, with provenance
recorded separately:

```text
PreparedRepresentation
  method
  method_version
  representation_format
  representation_version
  payload: Any
  metadata
```

Operations return explicit outcomes:

```text
prepare(image_path, image_metadata) -> PrepareOutcome
  representation
  method_internal_ms: optional
  diagnostics: dict

compare(representation_a, representation_b) -> CompareOutcome
  raw_score
  method_internal_ms: optional
  diagnostics: dict
```

The Python runner independently measures adapter wall time. An adapter can use
a Base64 string, array, object, or descriptor structure without changing the
general contract.

Every method declares non-empty `score_semantics` and one legal
`score_direction`:

```text
higher_is_more_similar
lower_is_more_similar
```

No normalization or thresholding is performed.

## Manifest preflight

Before warm-up or the first measured pair, the runner:

1. invokes the dedicated validator registered for the exact dataset/protocol;
2. rejects an empty manifest;
3. confirms every row has the expected dataset and protocol;
4. confirms the manifest path and SHA-256 match the run specification;
5. rejects CLI dataset/protocol values that disagree with manifest content.

The six registered validators are the existing SD300b/SD300c protocol
validators. Dataset images are read-only.

## Timing and warm-up

Execution is serial and `cold_pair`: for every result pair, preparation A,
preparation B, and comparison run in order. No representation cache is shared
between pairs, even when both image paths are equal.

The fixed warm-up policy executes the first manifest pair once (two prepares
and one comparison) before measured rows. Metadata records its pair ID, three
operations, and duration; it is not written to `pairs.csv`. SourceAFIS uses a
fresh managed JVM for each dataset/protocol, so warm/JIT state is never shared
between the six protocol runs.

Wall timings:

```text
prepare_a_ms
prepare_b_ms
compare_ms
total_ms
```

Optional method-internal timings:

```text
method_prepare_a_ms
method_prepare_b_ms
method_compare_ms
```

Wall time includes adapter/transport work. Method-internal time follows the
method's documented timing scope. Timings are finite and non-negative.

## Result schema v2

The exact schema identifier is `pairwise-result-v2`. Columns, in order, are:

```text
pair_id,dataset,protocol,subject_id,canonical_finger_position,method,method_version,benchmark_contract_version,result_schema_version,config_hash,implementation_hash,manifest_sha256,score_direction,score_semantics,raw_score,prepare_a_ms,prepare_b_ms,compare_ms,method_prepare_a_ms,method_prepare_b_ms,method_compare_ms,total_ms,prepare_a_diagnostics,prepare_b_diagnostics,compare_diagnostics,status,error_code,error_message
```

Valid status values are:

```text
ok
prepare_a_failure
prepare_b_failure
comparison_failure
```

`raw_score` is written with Python `repr(float_value)`, which is round-trip
safe. Historical v1 files used `.9g` (nine significant digits). Millisecond
fields use a shorter `.9g` representation.

Strict validation compares the result to the full ordered manifest, not just a
row count. It checks the exact unique pair-ID sequence; per-pair dataset,
protocol, subject, and canonical position; all run identity fields; legal
status; raw-score/error blankness; finite values; failure-stage field
blankness; and timing consistency. For success, within a 0.001 ms serialization
tolerance:

```text
total_ms >= prepare_a_ms + prepare_b_ms + compare_ms
```

## Safe bundle publication and reuse

The runner writes a sibling candidate directory, validates `pairs.csv` against
the manifest and run specification, builds metadata against the candidate
result SHA, validates the complete bundle, and only then atomically renames the
directory to its final path. Invalid candidates are removed. Existing bundles
are never overwritten. A failed directory promotion is rolled back when the
filesystem moved the candidate before reporting failure.

`--skip-existing` is gated by full bundle validation. Reuse is rejected when
manifest bytes/IDs, config, method version, implementation hash, contract
version, result SHA, or the metadata/result relationship changes—even if the
row count is unchanged.

## Reproducibility hashes

`result.sha256` covers the full CSV, including runtime timings, so independent
runs are not expected to have identical result-file hashes.

`score_payload_sha256` is SHA-256 over canonical JSON projecting only:

```text
pair_id
status
full-precision raw_score
error_code
```

It excludes timings and runtime environment. Equal score payload hashes are the
rerun reproducibility proof.

The deterministic `implementation_hash` covers the contract version, method
identity and score semantics, declared implementation provenance, sidecar JAR
SHA-256, Python adapter/client/lifecycle source SHA-256 values, benchmark
runner and support-source SHA-256 values, and benchmark contract source
SHA-256. Runtime durations, platform variability,
Java paths, timestamps, and Git dirty state are excluded. Full metadata still
records Java/runtime provenance and Git commit/dirty state when the workspace
is a Git checkout; otherwise it explicitly records that it is not one.

## Diagnostics

Deterministic diagnostics report per-run counts, zero/positive scores,
min/max/mean/median, zero-score pair IDs, and zero-score canonical-position
distribution. SD300b/SD300c comparisons join by protocol, subject, and
canonical position—not by row order—and report every score delta, exact
equality, mean/median absolute delta, Pearson correlation, and zero overlap.

SD300b and SD300c are resolution conditions over shared identities. They are
not treated as independent populations or combined to claim a doubled sample.
