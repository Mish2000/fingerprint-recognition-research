# DeepPrint-style FLX feasibility preflight v1

## Verdict

**NO-GO** for integration into `pairwise-benchmark-v2`.

The checkpoint is accessible, content-addressed, safely loadable, and technically
identified. Its state dictionary loads exactly into a `DeepPrint_TexMinu` model,
and source-defined SFinGe inference is deterministic on CPU. Integration is
nevertheless blocked by three frozen gates:

1. no explicit ownership or usage terms were found for the checkpoint;
2. the external source has no authoritative preprocessing path for an unknown
   SD300 image; and
3. it has no PPI-aware policy for applying one model to the 1000-PPI SD300b and
   2000-PPI SD300c images.

The frozen contract forbids inventing or selecting these policies from smoke
scores. Consequently, no SD300 image was passed to the model and no SD300
genuine, impostor, self, threshold, FAR, TAR, ROC, or AUC result was computed.

## Preflight identity

- starting target-repository commit:
  `45a57d80c36cba962b718e1b2d985779e1a6ccf7`
- contract: `research/deepprint_style_preflight_v1/preflight_contract.json`
- contract SHA-256:
  `d48354bc739bf46f2c7f5938ffee22cf0f05ae566d08f3d6d3baa79dd653227d`
- contract created before checkpoint download and before any SD300 inference
- external repository:
  [tim-rohwedder/fixed-length-fingerprint-extractors](https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors)
- frozen external commit:
  [`7accfca1f33b9b42bfd220e43cd5bc13b4a7fa13`](https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors/commit/7accfca1f33b9b42bfd220e43cd5bc13b4a7fa13)
- external tree: `e24273860264ab34e5503511cca6a0c798a23d8b`
- external checkout: clean detached HEAD
- tags: none; releases: none

## Original DeepPrint versus this candidate

The original work by Engelsma, Cao, and Jain is the DeepPrint research described
by the [original paper](https://arxiv.org/abs/1909.09901). The evaluated source is
a later FLX codebase used for the BIOSIG 2023 study by Rohwedder and colleagues.
Its README calls itself an implementation of DeepPrint and exposes several
smaller variants. It is not the authors' original implementation and this audit
does not establish exact reproduction of the 2019 paper.

Four identities therefore remain distinct:

1. original DeepPrint research;
2. the frozen FLX/BIOSIG-2023 reproduction source;
3. the single `best_model.pyt` artifact linked by that source; and
4. the model class proven by the artifact, `DeepPrint_TexMinu_512`.

The permitted name is **DeepPrint-style FLX/BIOSIG-2023 reproduction
candidate**.

## Source and licensing audit

The source tree is under the GNU Lesser General Public License version 3. The
repository's Inception v4 file includes a BSD 3-Clause notice for Remi Cadene's
implementation, and the ISO encoder/decoder directory contains an Apache 2.0
license. The relevant files and their SHA-256 values are recorded in
`external_source_manifest.json`. No LGPL or other external source was copied
into this repository.

The source license text contains notice, license-copy, source/relinking, and
replacement conditions that can matter when distributing a covered or combined
work. Most project-authored files have no per-file license header, so the
top-level license is repository-level evidence rather than a legal conclusion
about every file. This document describes facts and risk; it is not legal
advice.

### Checkpoint terms

The frozen README says only that a pretrained model with embedding size 512 is
available through its Google Drive link. Neither the source tree nor the
downloader-visible folder/file metadata states checkpoint ownership, an
academic-use grant, redistribution permission, commercial limitations, or
publication limitations. The README's paper-citation request is not treated as
a substitute for usage terms.

This unresolved status independently triggers the contract's NO-GO gate. A
written clarification from the artifact owner or repository maintainer is
required before integration.

## Checkpoint provenance and safety

- official source: Google Drive folder linked by the frozen README
- folder id: `1vV2skXApZMhqWTlF2j_qgXDxRYan5U1f`
- file id: `1R9v73hFBuy0iihz7v8eDlBcGrMDmQ9TH`
- filename: `best_model.pyt`
- size: 875,770,140 bytes
- SHA-256:
  `2683a04427bacd54adc00cfdc97474625b1e11e5a9e6672c5129f033018f8a28`
- format: PyTorch ZIP serialization archive
- archive members: 2,965; uncompressed bytes: 875,266,252
- folder contents: exactly one file
- stored outside the target repository

The artifact was loaded only in the isolated environment with:

```python
torch.load(checkpoint, map_location="cpu", weights_only=True)
```

The restricted load succeeded and returned a basic dictionary with
`model_state_dict`, `loss_state_dict`, and `optimizer_state_dict`. Unrestricted
unpickling was not used.

## Verified model identity

The README's “512” label is insufficient by itself because the source defines
multiple 512-dimensional variants. The state-dictionary names and shapes prove:

| Property | Verified value |
| --- | --- |
| class | `DeepPrint_TexMinu` |
| repository variant name | `DeepPrint_TexMinu_512` |
| training logits/classes | 8,000 |
| localization network | absent |
| Inception v4 grayscale stem | present |
| texture branch | present |
| minutiae branch | present |
| minutiae-map branch | present, training output only |
| texture embedding | 256 float32 values, L2-normalized |
| minutiae embedding | 256 float32 values, L2-normalized |
| matching representation | branch concatenation, 512 float32 values |
| representation norm | approximately `sqrt(2)` |
| PCA/dimensionality reduction | none in extraction path |
| learned reweighting | separate optional experiment; not part of checkpoint extraction |
| model parameters | 71,516,742 |

Strict loading used `DeepPrint_TexMinu(8000, 256, 256)` and produced:

```text
missing_keys = []
unexpected_keys = []
shape_mismatches = []
```

No training, fine-tuning, `strict=False`, or model choice based on SD300 was
required.

## Source preprocessing audit

The model input is one grayscale channel at 299x299. The source does not define
one general image-to-tensor function for an unknown dataset. Instead:

- SFinGe removes 32 bottom rows, converts to `[0,1]`, applies a Gabor
  binarizer with ridge width 5.0, square-pads, and antialiased-resizes;
- MCYT optical applies a fixed center crop and uses a 3.8-pixel binarizer;
- MCYT capacitive has a separate 4.8-pixel training binarizer, while a shared
  test helper applies the optical 3.8-pixel binarizer;
- FVC2004 uses a 1.8-pixel binarizer;
- NIST SD4 uses a 4.0-pixel binarizer; and
- fill values and crop/resize order vary by loader.

No mean/std standardization is used. Known paths convert an OpenCV grayscale
`uint8` image to a single-channel float tensor and use torchvision antialiased
resize. The dataset tutorial explicitly leaves crop/resize for a custom image
as a TODO. Thus there is no source-authoritative choice among inversion,
binarizer ridge width, crop, fill, and resize order for SD300.

Selecting one of these paths based on genuine scores would violate the frozen
contract. Selecting one without evidence would be arbitrary and also fails the
contract.

## PPI findings

SD300b images in the frozen cohort are 1000 PPI and SD300c images are 2000 PPI.
The FLX preprocessing functions take no PPI value. Their Gabor ridge widths are
fixed in pixels and selected by dataset, not physical ridge spacing. The model
has no localization network in this checkpoint, so it cannot be claimed to
normalize physical scale implicitly. Direct resize to 299x299 does not by
itself prove equal physical scale across scans.

The source contains no official resampling rule for mapping 1000 and 2000 PPI
to a common physical sampling density. A sidecar could implement such a policy
in a later, separately frozen task, but it must be justified before evaluation;
this preflight cannot choose it from smoke scores. The PPI gate fails.

## Training data and leakage analysis

| Dataset | Role reported by source | Subjects/fingers and impressions | Sensor | Public status in source | Direct SD300b/c overlap evidence | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| SFinGe | training | 6,000 fingers x 10 | synthetic | generator-based | none found | high for source code; medium for exact checkpoint |
| SFinGe separate subjects | validation | 2,000 fingers x 4 | synthetic | generator-based | none found | high for source code; medium for exact checkpoint |
| MCYT330 optical | first 2,000 training; last 1,300 test | 3,300 fingers x 12 | optical | described as public in paper summary | none found | high for source code; medium for exact checkpoint |
| MCYT330 capacitive | first 2,000 training; last 1,300 test | 3,300 fingers x 12 | capacitive | described as public in paper summary | none found | high for source code; medium for exact checkpoint |
| FVC2004 DB1A | loader/test code only | 100 fingers x 8 in code | optical | public benchmark | none found; source says not used in paper | medium |
| NIST SD4 | loader/test code only | 2,000 fingers x 2 in code | inked card scans | restricted/public status not resolved here | none found; source says not used in paper | medium |
| SD300b/c | target smoke only | frozen 10-finger, 40-image cohort | 1000/2000-PPI scans | local research data | no mention or loader found in FLX source | medium |

The two loss-center tensors and logits have 8,000 classes, consistent with the
source's 6,000 SFinGe plus 2,000 MCYT training identities. No evidence of direct
SD300 inclusion was found. However, the checkpoint carries no sample manifest,
training log, dataset hashes, or independent metadata proving exactly which
files produced it. Therefore there is no positive overlap evidence, but exact
checkpoint training provenance is still insufficient for a GO decision.

## Resolved isolated environment

The main repository environment was not modified. The isolated CPU environment
is recorded in `environment_resolved.yml` and `pip_freeze.txt`.

| Component | Version |
| --- | --- |
| Python | 3.9.23 |
| PyTorch | 2.5.1+cpu |
| torchvision | 0.20.1+cpu |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 / runtime 4.10.0 |
| SciPy | 1.13.1 |
| scikit-learn | 1.5.2 |
| torchmetrics | 1.6.1 |

CPU execution used one PyTorch intra-op thread, one inter-op thread, one OpenCV
thread, fixed Python/NumPy/PyTorch/OpenCV seeds, `model.eval()`,
`torch.inference_mode()`, and deterministic PyTorch algorithms. CUDA was not
used or required.

## CPU inference result

No SD300 inference was run. To separate checkpoint viability from SD300 policy,
two source-provided SFinGe images were processed through the source-defined
SFinGe test path. The checkpoint produced finite, non-constant 512-dimensional
float32 representations. Each 256-dimensional branch had norm approximately
1, so concatenated norms were approximately `sqrt(2)`.

The two representation hashes were:

```text
f164860e19a9f24bb5a116e82e97a87983173609356960e706e1a447d4c2b866
aca69dc2c3ee5bcb945ee81b3b8fd4ed97e6ce55bf2486b881ce6385ec84e470
```

Three new processes reproduced both preprocessing tensor hashes, both
representation hashes, shapes, dtypes, finite status, and scores exactly. This
is a technical checkpoint/environment result only; it is not SD300 validity or
accuracy evidence.

## Frozen SD300 smoke cohort and unrun gates

The cohort was frozen before inference in `smoke_cohort.json`:

```text
10 identities x 2 roles x 2 datasets = 40 images
cohort SHA-256 = 3c9d26c6e85842ed25d53f6823a986d2bef518163a42a3868cc2d762895e95ef
```

All 40 source images existed and were content-hashed. None was replaced. Because
preprocessing/PPI and checkpoint-license gates failed before model input:

- images attempted: 0
- valid images: 0
- technical failures: 0
- images not attempted: 40
- SD300 embedding shape/dtype/finite/norm/hash checks: not run
- six-image, three-process SD300 repeatability: not run
- SD300 self/genuine/impostor diagnostics: not run
- threshold/FAR/TAR/ROC/AUC: not computed

`smoke_report.json` and `repeatability_report.json` preserve those `not_run`
statuses and the exact blockers.

## Similarity audit

`CosineSimilarityMatcher.similarity` is named as cosine similarity but computes
a direct NumPy dot product. Extraction L2-normalizes each branch independently,
then experiment loading concatenates texture and minutiae embeddings. Therefore
the verified unreweighted comparison is:

```text
score(a, b) = dot(texture_a, texture_b) + dot(minutiae_a, minutiae_b)
```

Higher is more similar. With two unit-norm branches, the expected self score is
2 (subject to float32 rounding), and the finite theoretical range is `[-2, 2]`.
The source's vectorized matcher clips negative scores to zero, whereas its scalar
matcher does not; a future adapter must use and freeze one authoritative scalar
formula. This preflight records the scalar formula above and did not select it
from SD300 results. No reweighting, score normalization, PCA, or threshold is
part of this representation.

## Future integration design note

If and only if the failed gates are resolved, use an **isolated sidecar** rather
than in-process integration. The source targets Python 3.9+, requires a large
PyTorch runtime, and uses an 876 MB externally stored checkpoint. A sidecar keeps
those dependencies and weights out of the Python 3.11 research environment and
supports process-level provenance and repeatability.

Proposed contract:

```text
prepare(image_path, image_metadata)
  -> versioned fixed-length representation

compare(representation_a, representation_b)
  -> raw higher-is-more-similar dot-product score
```

The future representation should contain a canonical header plus 512 contiguous
little-endian float32 values ordered as texture[256], minutiae[256]. Its version
must bind the external commit, checkpoint SHA-256, verified model class and
dimensions, preprocessing/PPI policy, dependency versions, deterministic flags,
and implementation-component hashes. The final `config_identity` cannot be
computed until preprocessing and PPI policy are legitimately resolved.

Recommended operational details:

- serialization: canonical JSON header plus length-delimited binary float32
  payload, or an equivalently specified container;
- score direction: higher is more similar;
- runtime provenance: source commit, checkpoint hash, Python/PyTorch/
  torchvision/NumPy/OpenCV versions, CPU/GPU mode, thread counts, and seeds;
- failure codes: `image_read_failure`, `unsupported_image`,
  `preprocessing_policy_undefined`, `ppi_policy_undefined`,
  `checkpoint_integrity_failure`, `model_state_mismatch`, `inference_failure`,
  and `non_finite_embedding`;
- cache key: image SHA-256 plus complete config identity; no raw images,
  embeddings, or weights in Git;
- CPU policy: CPU is the release/repeatability gate;
- GPU policy: separately versioned and validated later, never implicitly mixed;
- threshold policy: compare returns only the raw score; calibration and
  thresholds remain a separate task.

No adapter is implemented by this preflight.

## Exact GO-gate results

| # | Frozen GO gate | Result | Evidence |
| ---: | --- | --- | --- |
| 1 | external commit fixed | pass | commit and tree recorded; detached clean checkout |
| 2 | relevant source available | pass | source and notebooks audited |
| 3 | source license documented | pass | LGPL-3.0 plus BSD/Apache components |
| 4 | checkpoint use status clear enough | **fail** | no explicit checkpoint terms |
| 5 | checkpoint from official source | pass | single file downloaded from README link |
| 6 | checkpoint hash fixed | pass | SHA-256 recorded |
| 7 | variant identified from source/state | pass | `DeepPrint_TexMinu_512` proven by keys/shapes |
| 8 | exact state-dictionary load | pass | strict load, no missing/unexpected/shape mismatch |
| 9 | no training/fine-tuning | pass | inference checkpoint loads directly |
| 10 | training data sufficiently documented | **fail** | source-level split known; exact artifact sample provenance absent |
| 11 | no direct SD300 overlap evidence | pass with limited confidence | no SD300 code/metadata found; exact provenance incomplete |
| 12 | one deterministic preprocessing policy | **fail** | only dataset-specific paths; no SD300 authority |
| 13 | Plain and Roll preprocessing | not run | stopped at failed policy gate |
| 14 | SD300b and SD300c preprocessing | not run | stopped at failed policy/PPI gate |
| 15 | finite, non-constant SD300 embeddings | not run | no SD300 inference; external examples passed technically |
| 16 | fresh-process SD300 repeatability | not run | no SD300 inference; external examples were byte-identical |
| 17 | similarity identified independently | pass | source scalar dot-product path audited |
| 18 | benchmark contract need not change | pass | sidecar design fits prepare/compare contract |
| 19 | practical integration architecture | pass conditionally | isolated sidecar design exists |
| 20 | no unresolved security/provenance issue | **fail** | checkpoint usage and exact training provenance unresolved |

Additional frozen NO-GO conditions triggered:

- checkpoint license/usage status is insufficiently clear;
- SD300 preprocessing requires an arbitrary choice; and
- no explained 1000/2000-PPI policy exists.

Technical safety checks did not find a state-dictionary or restricted-loading
failure. They do not override the failed gates.

## Unresolved risks

- checkpoint ownership and permitted academic/commercial/redistribution use;
- exact sample-level training and validation provenance for this artifact;
- physically justified preprocessing for SD300 plain and rolled impressions;
- a source-authoritative or independently justified 1000/2000-PPI policy;
- scalar-versus-vectorized negative-score behavior;
- dependency drift because the external `requirements.txt` is entirely
  unbounded; and
- LGPL/BSD/Apache compliance details for any future distributed sidecar.

## Next task

**Proceed to a Gabor/FingerCode full-system feasibility and implementation
task.**

Do not continue to a DeepPrint-style adapter unless a future task first resolves
the checkpoint terms and freezes a justified preprocessing/PPI contract without
using evaluation scores for selection.
