# Fingerprint Recognition Research

This repository contains read-only discovery and deterministic pairwise
benchmarking for the local NIST SD300b (1000 PPI) and SD300c (2000 PPI)
datasets. Six genuine-pair manifests are maintained:

- `plain_self`, `roll_self`, and `plain_roll` for SD300b;
- `plain_self`, `roll_self`, and `plain_roll` for SD300c.

The historical SourceAFIS v1 results under
`results/<dataset>/<protocol>/sourceafis/` are preserved as an immutable
baseline. Hardened runs use `pairwise-benchmark-v2` directories below those
locations.

## Dataset layout

The code expects:

```text
C:\fingerprint-datasets\NIST\sd300b\images\1000\png\plain
C:\fingerprint-datasets\NIST\sd300b\images\1000\png\roll
C:\fingerprint-datasets\NIST\sd300c\images\2000\png\plain
C:\fingerprint-datasets\NIST\sd300c\images\2000\png\roll
```

Dataset files are inputs only and must not be modified.

## Environment

```powershell
conda env create -f environment.yml
conda activate fingerprint-recognition-research
python -m pip install -e .
```

Run the complete Python suite:

```powershell
python -m pytest
```

The pinned Python SIFT dependency is `opencv-python==4.12.0.88`; the headless
distribution must not be installed alongside it.

Build and test the SourceAFIS sidecar:

```powershell
mvn -f apps\sourceafis-sidecar\pom.xml test
mvn -f apps\sourceafis-sidecar\pom.xml package
```

## Manifest validation

Each benchmark run invokes its dedicated validator automatically. Validators
can also be run directly, for example:

```powershell
fingerprint-sd300b-plain-self validate --data-root C:\fingerprint-datasets --manifest protocols\sd300b\plain_self.csv
fingerprint-sd300c-plain-roll validate --data-root C:\fingerprint-datasets --manifest protocols\sd300c\plain_roll.csv
```

The fixed manifest schema is:

```text
pair_id,dataset,protocol,subject_id,canonical_finger_position,ppi,raw_frgp_a,raw_frgp_b,path_a,path_b
```

## SourceAFIS benchmark v2

The default command starts a fresh managed SourceAFIS JVM for each of the six
runs, applies the same deterministic warm-up, and publishes validated bundles
atomically:

```powershell
python -m fingerprint_benchmark.cli run-sourceafis-all --sidecar-jar apps\sourceafis-sidecar\target\sourceafis-sidecar-0.2.0.jar --data-root C:\fingerprint-datasets
```

Safe reuse performs full bundle validation:

```powershell
python -m fingerprint_benchmark.cli run-sourceafis-all --skip-existing
```

One run can be executed with `run-sourceafis --dataset ... --protocol ...`.
Use a different `--results-root` for a reproducibility rerun so the primary
bundle is not overwritten.

After all six primary runs, create score, paired SD300b/C, and v1 comparison
reports:

```powershell
python -m fingerprint_benchmark.cli diagnose-sourceafis-v2
```

Contract details are in [docs/benchmark_contract.md](docs/benchmark_contract.md)
and SourceAFIS timing/lifecycle details are in
[docs/sourceafis_integration_v2.md](docs/sourceafis_integration_v2.md).

## SIFT geometric baseline

The single public SIFT method is `sift_geometric`, version
`sift-geometric-v1`. Its development protocol, exact OpenCV parameters,
matching, geometry, score semantics, leakage controls, and cohort rule are in
[docs/sift_geometric.md](docs/sift_geometric.md).

Run the leakage-safe development stages explicitly:

```powershell
fingerprint-sift-study prepare
fingerprint-sift-study parity
fingerprint-sift-study pilot
```

After the pilot freezes `sift_geometric_config.json` and the decision rule,
run the six original manifests and build the SIFT-specific reports:

```powershell
fingerprint-sift-study run
fingerprint-sift-study report
```

## SourceAFIS reproducibility audit

The isolated, pre-specified audit workflow is documented in
[docs/sourceafis_reproducibility_audit.md](docs/sourceafis_reproducibility_audit.md).
Only its explicit `run` phase invokes SourceAFIS; `prepare` and `compare` are
read-only with respect to primary benchmark artifacts and datasets.

## Per-dataset SourceAFIS self-accept protocol

The method-specific threshold-40 derived protocol is implemented by
`fingerprint-derived-sourceafis` (or
`python -m fingerprint_benchmark.derived_protocol`). Its guarded phases are
`prepare`, `preflight`, `run`, `report`, and `integrity`. Derived manifests and
runs are published below `results/derived_protocols/` and
`results/derived_protocol_runs/`; the six primary bundles are never replaced.

## Per-dataset SIFT self-accept protocol

The frozen SIFT protocol is implemented by `fingerprint-derived-sift` (or
`python -m fingerprint_benchmark.sift_derived_protocol`). Its guarded phases
are `protect-before`, `prepare`, `preflight`, `run`, `report`, and
`protect-after`. The mandatory `preflight` compares 30 uniformly spaced pairs
per dataset exactly with their primary results; any mismatch forbids the two
full derived runs. Existing SIFT, SourceAFIS, manifest, and dataset artifacts
are read-only.

## Shared SourceAFIS/SIFT biometric accuracy

`fingerprint-shared-accuracy` implements the separate prepared-representation
accuracy study under `results/shared_accuracy/sourceafis_sift_v1/`. It compares
both methods on identical subjects, genuine pairs, logical impostor pairs, and
development-calibrated FAR targets of 1% and 0.1%. It does not modify or replace
the cold-pair timing benchmark. The staged safety gates, cache scope,
calibration rule, confidence reporting, and execution commands are documented
in [docs/shared_biometric_accuracy_protocol.md](docs/shared_biometric_accuracy_protocol.md).
