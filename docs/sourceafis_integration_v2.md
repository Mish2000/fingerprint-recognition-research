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
java -jar apps\sourceafis-sidecar\target\sourceafis-sidecar-0.3.0.jar
```

Run through the benchmark CLI with a dedicated JVM for each dataset/protocol run:

```powershell
python -m fingerprint_benchmark.cli run-sourceafis-all --sidecar-jar apps\sourceafis-sidecar\target\sourceafis-sidecar-0.3.0.jar
```

## Sidecar Scope

The sidecar supports only:

```text
GET  /health
POST /extract-template
POST /extract-final-minutiae
POST /verify
```

It does not implement `/identify`, resident galleries, candidate search,
threshold calibration, decision logic, fusion, dataset policy, CSV writing,
manifest parsing, score normalization, or preprocessing.

The sidecar is stateless. It does not persist templates or images and must not
log image bytes, template bytes, or base64 payloads.

## Process Lifecycle

Sidecar contract `sourceafis-sidecar-v2.2` is implemented by artifact version
`0.3.0`. SourceAFIS remains pinned to `3.18.1`. A dedicated JVM starts before
each dataset/protocol run and remains
alive only for that run. There is no subprocess per pair, no JVM startup inside
`prepare`, and no JVM startup inside `compare`. The same deterministic warm-up
policy runs before measured pairs for every dataset/protocol run. Warm-up
operations are not result rows.

Startup validation is performed before pair timing starts. Pair timings include
per-operation HTTP transport overhead. They exclude sidecar startup, startup
health validation, and shutdown.

The Python client reuses a persistent HTTP connection for the run and is closed
at the end.

For the SourceAFIS-location joint-500 branch, orchestration starts one fresh JVM
per result bundle, keeps one validated persistent client within that bundle,
and does not cache representations between pairs. Detector and adapter do not
close a client they did not create. The orchestration layer owns both client and
sidecar lifecycle.

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

For `/extract-final-minutiae`, method-internal time includes options and raw
image construction, feature extraction, `toByteArray()`, documented native
CBOR parsing, and response-model construction. It excludes HTTP, JSON, request
Base64 decoding, and response JSON serialization. Detector wall time separately
records pixel serialization, complete sidecar request wall time, SourceAFIS
method-internal time, coordinate mapping/sorting, and total detector time.

## Exact raw-pixel endpoint

`POST /extract-final-minutiae` accepts:

```json
{
  "width": 1234,
  "height": 1600,
  "pixels_base64": "...",
  "dpi": 1000
}
```

`pixels_base64` decodes to exactly `width * height` bytes of uint8 grayscale,
row-major from top-left to bottom-right (`0` black, `255` white). The sidecar
passes those bytes directly to the public raw `FingerprintImage` constructor.
There is no PNG/JPEG round trip, ImageIO conversion, external resize,
enhancement, inversion, binarization, normalization, or silent DPI default.

The complete response schema is:

```json
{
  "sourceafis_version": "3.18.1",
  "template_version": "3.18.1-java",
  "effective_dpi": 1000.0,
  "native_width": 1234,
  "native_height": 1600,
  "scaled_width": 617,
  "scaled_height": 800,
  "coordinate_space": "sourceafis_500_dpi_scaled_image",
  "selection_stage": "sourceafis_final_template_minutiae",
  "selection_semantics": "sourceafis_final_selected_minutia_set",
  "source_order_semantics": "deterministic_sourceafis_template_order_not_quality_ranking",
  "template_sha256": "64 lowercase hexadecimal characters",
  "minutia_count": 47,
  "minutiae": [
    {
      "source_index": 0,
      "x_scaled": 74,
      "y_scaled": 136,
      "direction_radians": 1.9513026,
      "type": "ENDING"
    }
  ],
  "method_internal_ms": 12.34
}
```

The endpoint never returns serialized template bytes, a score, threshold, or
decision. Parser/schema failures return explicit error codes rather than a
partial response.

## Native-template parsing

Production creates `FingerprintTemplate`, serializes it with `toByteArray()`,
and parses only the [documented SourceAFIS native CBOR format](https://sourceafis.machinezoo.com/template):
`version`, scaled `width`/`height`, integer `positionsX`/`positionsY`, float32
`directions`, and the `E`/`B` type string. Root type, exact fields, lengths,
version family, positive dimensions, coordinates, finite directions, types,
and counts are validated. No reflection, private access, Java-object parsing,
or production algorithm transparency is used.

A Java-only regression test activates a custom `FingerprintTransparency`
consumer that accepts only `top-minutiae`. It canonicalizes that set by X, Y,
type, and exact float32 direction bits and proves set equality against the
native final template for the synthetic test images. Order is deliberately not
compared.

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

Health additionally reports the v2.2 final-minutiae capability, endpoint, raw
input contract, scaled coordinate space, final-template stage, and exact timing
scope. The Python client validates every field before a run.
