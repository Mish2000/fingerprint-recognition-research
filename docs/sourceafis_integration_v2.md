# SourceAFIS Integration v2

This repository uses the official maintained SourceAFIS Java implementation as
the first baseline:

```text
com.machinezoo.sourceafis:sourceafis:3.18.1
```

The version is pinned in `apps/sourceafis-sidecar/pom.xml`. Python does not link
directly to SourceAFIS. The architecture is:

```text
Python benchmark runner -> narrow SourceAFIS adapter/client -> local Java sidecar -> official SourceAFIS Java
```

## Runtime Requirements

- Java runtime capable of running the shaded sidecar jar.
- Maven for building and testing the sidecar.
- Localhost HTTP transport only for this milestone.

Build and test:

```powershell
cd apps\sourceafis-sidecar
mvn test
mvn package
```

Run manually:

```powershell
$env:SOURCEAFIS_HOST = "127.0.0.1"
$env:SOURCEAFIS_PORT = "8765"
java -jar apps\sourceafis-sidecar\target\sourceafis-sidecar-0.2.0.jar
```

Run through the benchmark CLI with a dedicated JVM for each dataset/protocol run:

```powershell
python -m fingerprint_benchmark.cli run-sourceafis-all --sidecar-jar apps\sourceafis-sidecar\target\sourceafis-sidecar-0.2.0.jar
```

## Sidecar Scope

The sidecar supports only:

```text
GET  /health
POST /extract-template
POST /verify
```

It does not implement `/identify`, resident galleries, candidate search,
threshold calibration, decision logic, fusion, dataset policy, CSV writing,
manifest parsing, score normalization, or preprocessing.

The sidecar is stateless. It does not persist templates or images and must not
log image bytes, template bytes, or base64 payloads.

## Process Lifecycle

Sidecar contract `sourceafis-sidecar-v2.1` is implemented by artifact version
`0.2.0`. A dedicated JVM starts before each dataset/protocol run and remains
alive only for that run. There is no subprocess per pair, no JVM startup inside
`prepare`, and no JVM startup inside `compare`. The same deterministic warm-up
policy runs before measured pairs for every dataset/protocol run. Warm-up
operations are not result rows.

Startup validation is performed before pair timing starts. Pair timings include
per-operation HTTP transport overhead. They exclude sidecar startup, startup
health validation, and shutdown.

The Python client reuses a persistent HTTP connection for the run and is closed
at the end.

The client and managed sidecar accept only the explicit loopback hosts
`localhost`, `127.0.0.1`, and `::1`. The Java service also refuses any other
bind host. Biometric images and templates are never sent over plain HTTP to a
remote network.

## Timing Decomposition

Python records adapter wall time around each complete operation. It includes
image reading, request/response Base64 work, JSON, HTTP transport, and the Java
operation. The sidecar additionally returns `method_internal_ms`, measured with
`System.nanoTime()`.

For `/extract-template`, method-internal time includes
`FingerprintImageOptions` construction, `FingerprintImage` construction,
`FingerprintTemplate` extraction, and `FingerprintTemplate.toByteArray`
serialization. It excludes HTTP, JSON parsing, request Base64 decoding, and
response Base64 encoding.

For `/verify`, method-internal time includes deserialization of both
`FingerprintTemplate` objects, `FingerprintMatcher` construction, and
`FingerprintMatcher.match`. It excludes HTTP, JSON parsing, and request Base64
decoding. It is therefore not described as pure matcher time.

## DPI/PPI Policy

DPI/PPI is required. The adapter derives effective DPI from manifest/prepare
metadata, preferring `ppi` and accepting `dpi` only as a secondary key.

Policy:

- Missing DPI/PPI fails explicitly.
- Non-numeric, NaN, infinity, negative, or out-of-range DPI/PPI fails
  explicitly.
- Valid range is 100 to 4000 DPI.
- There is no hidden fallback to 500 DPI.
- There is no external resize or resampling before SourceAFIS.

SD300b passes 1000 PPI. SD300c passes 2000 PPI.

## Raw Score Semantics

`raw_score` is the unnormalized SourceAFIS similarity score returned by
`FingerprintMatcher.match`. The adapter does not return a probability,
confidence, threshold, decision, sigmoid output, min-max normalized value, or
other calibrated score.

## Template Compatibility

Serialized SourceAFIS templates are version-sensitive. Every representation
records method, SourceAFIS version, representation format, representation
version, and effective DPI. Image hashing and other provenance enrichment are
not performed inside timed preparation. This milestone does not add a
persistent template cache.

## Licensing And Provenance

SourceAFIS is consumed only through the official Maven artifact above. The
sidecar build metadata reports the Maven coordinates, pinned SourceAFIS version,
sidecar contract version, sidecar implementation version, and actual Java
runtime version/vendor. Managed startup metadata additionally records the
resolved Java executable, exact command, sidecar JAR path, and sidecar JAR
SHA-256.
