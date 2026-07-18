# Pairwise Benchmark Contract v2

The repository's method-neutral benchmark contract is identified by:

```text
pairwise-benchmark-v2
```

It defines serial evaluation of an ordered pair manifest and publication of
raw pairwise scores. Method-specific detection, representation, and comparison
behavior belongs to adapters such as the OpenCV Harris detector-only adapter
and the external SourceAFIS adapter.

## Adapter contract

Every adapter declares its method identity, version, configuration, score
semantics, score direction, runtime information, and implementation
provenance. The two operations are:

```text
prepare(image_path, image_metadata) -> PrepareOutcome
compare(representation_a, representation_b) -> CompareOutcome
```

`PreparedRepresentation.payload` is opaque to the runner. It can contain an
array, object, descriptor structure, or external-system template. The runner
requires the representation's method and version to match the active adapter.

`PrepareOutcome` and `CompareOutcome` carry optional method-internal timing and
structured diagnostics. A successful comparison returns one finite raw score.
The adapter declares whether higher or lower scores are more similar. The
benchmark performs no thresholding, calibration, score normalization, or
decision making.

## Immutable run identity

The run specification binds:

- expected dataset and protocol;
- resolved manifest path and its SHA-256;
- method name and version;
- benchmark contract version;
- canonical configuration hash;
- implementation hash.

`config_hash` is SHA-256 over canonical JSON containing the adapter
configuration, method identity, score semantics, timing mode, warm-up policy,
and contract version. `implementation_hash` covers stable code and dependency
provenance that can affect the result.

## Manifest validation

Before warm-up or measured execution, preflight validation:

1. invokes the validator registered for the requested dataset and protocol;
2. rejects an empty manifest;
3. verifies every row's dataset and protocol;
4. checks the resolved manifest path and SHA-256 against the run specification;
5. preserves the manifest's exact, unique pair-ID order.

Dataset images and protocol manifests are treated as read-only inputs.

## Pair execution and result rows

Execution is serial. For every measured pair, the runner performs preparation
A, preparation B, and comparison in that order. The first manifest pair is
executed once as warm-up and recorded only in run metadata.

Each result row records:

- pair, dataset, protocol, subject, and canonical-finger identity;
- method, contract, configuration, implementation, and manifest hashes;
- raw score and declared score semantics;
- wall-clock and optional method-internal timings;
- structured preparation/comparison diagnostics;
- explicit success or failure status, error code, and error message.

Legal statuses are `ok`, `prepare_a_failure`, `prepare_b_failure`, and
`comparison_failure`. Success rows contain a finite raw score and no error.
Failure rows preserve the exact failing stage and do not synthesize a score.

Wall timings are finite, non-negative milliseconds:

```text
prepare_a_ms
prepare_b_ms
compare_ms
total_ms
```

The serialized raw score uses Python's round-trip-safe `repr(float_value)`.

## Bundle publication

Bundles use the contract-relative layout:

```text
<results-root>/<dataset>/<protocol>/<method>/pairwise-benchmark-v2/<config_hash>/
```

Each bundle contains canonical `pairs.csv` and `run_metadata.json` artifacts.
The runner writes a sibling candidate directory, validates its result rows
against the full ordered manifest and run specification, builds metadata,
validates the complete bundle, and only then promotes it atomically. Invalid
candidates are removed, and existing valid bundles are never overwritten.

`--skip-existing` reuses a bundle only after the same full validation. Reuse is
rejected when manifest bytes or IDs, method identity, configuration,
implementation, contract version, result hash, or metadata relationships do
not match.

## Reproducibility hashes

`result.sha256` covers the full CSV, including timings. Independent executions
therefore need not have the same full-file hash.

`score_payload_sha256` hashes canonical JSON containing only:

```text
pair_id
status
full-precision raw_score
error_code
```

This excludes runtime timings and environment variability and is the compact
proof for exact pairwise-score reproduction.

The deterministic implementation hash includes the contract and method
identity, score semantics, adapter-declared source hashes, benchmark runner and
support-source hashes, and method-specific stable dependency hashes. Runtime
durations, timestamps, platform paths, and Git dirty state are excluded from
that hash while remaining available in full run metadata.
