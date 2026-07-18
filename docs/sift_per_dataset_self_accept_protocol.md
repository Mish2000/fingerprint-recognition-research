# SIFT per-dataset self-accept derived protocol

This workflow aligns the existing full SIFT results with the per-method,
per-dataset protocol. It never reruns self comparisons, changes a threshold,
retunes a SIFT parameter, or uses a plain-roll outcome to select membership.

For each dataset independently, an anatomical identity is included only when
its frozen `plain_self` row has `status == ok` and is accepted, its frozen
`roll_self` row has `status == ok` and is accepted, and the base `plain_roll`
manifest contains that identity. Identity is exactly
`(subject_id, canonical_finger_position)`. The frozen decision is
`raw_score >= primary_threshold` from `decision_rule.json` (currently 4.0 for
both datasets).

The immutable provenance gates require these file SHA-256 values:

- SIFT configuration: `f9f0623ae89752d09c5933d49dc80acc5803863cc8dc7109efb98b96d282f01f`
- Decision rule: `13e9e29d918f95783d68eecb70f6aa857009ac902417d3ac8d59dcf59b7a98fa`

Run phases in order:

1. `protect-before` hashes every file in the two in-scope trees
   `NIST/sd300b` and `NIST/sd300c`, plus all protected manifests, SourceAFIS
   artifacts, SIFT primary/development artifacts, and execution sources.
2. `prepare` validates all six primary SIFT bundles and atomically publishes
   the two byte-exact derived manifest subsets plus inclusion/exclusion
   provenance.
3. `preflight` selects exactly 30 positions per dataset with
   `floor(i * N / 30)`, runs the frozen cold-pair adapter, and requires exact
   score text, numeric score, decision, status, error code, and deterministic
   diagnostics equality. Timing is excluded. A failure is terminal and the
   full runs are forbidden.
4. `run` executes only the two derived `plain_roll` manifests when both
   preflight conditions passed. On Windows, `--execution-results-root` may be
   the same physical run directory written with the official extended-length
   `\\?\` prefix (or a short alias), preventing MAX_PATH failures without
   changing artifact location or provenance.
5. `report` retains accepted, rejected, and failed plain-roll rows, compares
   every rerun pair with its primary row by `pair_id`, and publishes supervisor
   and SourceAFIS-alignment reports.
6. `protect-after` rehashes the complete protected inventory, requires exact
   before/after equality, and publishes an output artifact manifest.

Derived manifests live under
`results/derived_protocols/sift_geometric_per_dataset_self_accept_v1/` and
derived runs under
`results/derived_protocol_runs/sift_geometric_per_dataset_self_accept_v1/`.
No file below the primary protocol or primary method namespaces is replaced.
