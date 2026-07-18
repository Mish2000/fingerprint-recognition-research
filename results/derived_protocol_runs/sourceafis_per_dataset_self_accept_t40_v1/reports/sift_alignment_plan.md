# SIFT alignment plan

No SIFT code or existing SIFT artifact was changed, and no SIFT rerun was performed.

## Reusable artifacts

- Frozen configuration: `C:\fingerprint-recognition-research\results\sift_geometric\development\sift_geometric_config.json` (SHA-256 `f9f0623ae89752d09c5933d49dc80acc5803863cc8dc7109efb98b96d282f01f`).
- Frozen decision rule: `C:\fingerprint-recognition-research\results\sift_geometric\development\decision_rule.json` (SHA-256 `13e9e29d918f95783d68eecb70f6aa857009ac902417d3ac8d59dcf59b7a98fa`).
- All four full `plain_self`/`roll_self` result bundles.
- Both full `plain_roll` result bundles for later exact-score comparison.

## Required alignment work (not executed)

1. Validate the six existing SIFT benchmark bundles and frozen decision rule.
2. For SD300b only, select identities accepted by SD300b `plain_self` and SD300b `roll_self`, then intersect with SD300b `plain_roll` availability.
3. Independently repeat the same operation for SD300c using its frozen threshold.
4. Publish two method- and dataset-specific byte-exact derived manifests with inclusion/exclusion provenance.
5. Rerun only those two derived `plain_roll` manifests under the frozen SIFT configuration.
6. Compare rerun scores exactly to the corresponding full primary SIFT rows and publish decision/timing tables.

The existing joint cross-dataset SIFT cohort must not be reused as the new evaluation population, because it imposes cross-dataset membership equality.
