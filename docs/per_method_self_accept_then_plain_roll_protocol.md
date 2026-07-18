# Per-method self-accept then plain-roll protocol

Protocol namespace implemented for SourceAFIS: `sourceafis_per_dataset_self_accept_t40_v1`.

The protocol is method-specific and dataset-specific. Its identity key is
`(subject_id, canonical_finger_position)`. A cohort produced by one method is
never reused by another method.

## Frozen decision rule

Before any evaluation selection, the method's self-test decision rule must be
frozen without looking at `plain_roll` outcomes. For the SourceAFIS
implementation in this repository, the frozen threshold is `40.0`, and a self
row passes exactly when `status == "ok"` and `raw_score >= 40.0`.

The `plain_roll` score, status, timing, or diagnostics must never influence
whether its identity is included.

## Required workflow for every future method

1. Treat SD300b and SD300c as separate datasets. Do not require their selected
   identity sets or counts to be equal.
2. Run the method's `plain_self` and `roll_self` benchmarks on the immutable
   base manifests, preserving the full primary results.
3. Freeze the method configuration and decision rule using development data or
   another pre-specified leakage-safe procedure. Do not calibrate on the
   `plain_roll` evaluation results.
4. Within each dataset, select identities that pass both the method's
   `plain_self` and `roll_self` decisions.
5. Intersect that set with identities available in the dataset's immutable
   base `plain_roll` manifest.
6. Publish a method- and dataset-specific derived `plain_roll` manifest. The
   manifest must retain the exact base schema, source row contents, and source
   row order; only row inclusion may change.
7. Rerun the method only on the two derived `plain_roll` manifests under the
   same frozen implementation, configuration, PPI policy, timing contract, and
   cold-pair lifecycle used for the primary results.
8. Preserve both the full primary bundles and the derived bundles in separate,
   immutable namespaces with complete SHA-256 provenance.
9. Never use a cohort selected by one algorithm for another algorithm. Each
   method receives its own self-accepted identities and derived manifests.
10. Never remove a derived pair because its `plain_roll` result is rejected.
    That rejection is an experimental false non-match and remains in all
    summaries.

## Required validation and provenance

Every implementation must validate the complete primary bundles and base
manifests before selection, record inclusion and all applicable exclusion
reasons, prove completeness of the intersection, publish atomically, and
compare rerun status, error code, deterministic diagnostics, and raw score to
the corresponding primary row by `pair_id`. Timing equality is not required.

Protected manifests, primary bundles, audits, cohorts, existing method
artifacts, and dataset files must have matching before/after content hashes.
