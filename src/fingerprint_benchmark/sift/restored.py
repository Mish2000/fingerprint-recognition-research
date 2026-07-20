"""Restoration layer binding the historical SIFT method to the current runner.

This module contains no algorithm.  Detection, description, matching, geometry
and scoring live unchanged in the sibling modules restored byte-for-byte from
the historical source commit; see ``docs/sift_geometric_full.md``.

Two narrow adaptations are required because the surrounding architecture moved
on after the historical bundles were produced:

1.  The historical runs read their frozen configuration from a development
    artifact under ``results/``.  Result artifacts are no longer tracked, so the
    same frozen values are declared here as code and covered by parity tests.
2.  ``provenance.implementation_provenance`` used to hash the ``sift`` package
    through a hardcoded method branch.  That branch was replaced by the generic
    ``ImplementationSourceProvider`` capability, so the adapter now declares its
    own sources instead of being special-cased.

Neither adaptation touches score semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .adapter import SiftGeometricAdapter
from .config import METHOD_NAME, METHOD_VERSION, SiftGeometricConfig


#: Commit whose ``src/fingerprint_benchmark/sift`` tree produced the historical
#: bundles.  Every restored module hashes to the value that commit's artifacts
#: recorded in ``implementation_hash_components.method_support_source_sha256``.
HISTORICAL_SOURCE_COMMIT = "e95ab6f3c5685df1222f32b7260b2e340e6b0d5e"

#: ``config_hash`` carried by all six historical bundles.  The restored adapter
#: is expected to reproduce this exactly; it is the frozen algorithm identity.
HISTORICAL_CONFIG_HASH = "51cd949cbfd84eebfe7cc5057f3621d6ec20be855905db743f42e8945adc541a"

#: ``implementation_hash`` of the historical bundles.  This is deliberately NOT
#: expected to reproduce: it mixes in support-module hashes and the removed
#: method branch, both of which changed after the runs.  It is recorded so the
#: restoration can state precisely what it does and does not reproduce.
HISTORICAL_IMPLEMENTATION_HASH = "a663786cf6503cb522ab0194c1ab6bc65698d366408c53d49defce09e4960bbb"

#: Files restored byte-for-byte from :data:`HISTORICAL_SOURCE_COMMIT`.
HISTORICAL_MODULE_FILENAMES = (
    "__init__.py",
    "adapter.py",
    "config.py",
    "descriptors.py",
    "extractor.py",
    "geometry.py",
    "matching.py",
    "preprocessing.py",
    "scoring.py",
)

#: SHA-256 of each restored module as recorded by the historical artifacts.
HISTORICAL_MODULE_SHA256 = {
    "__init__.py": "960b1b5110495fdbe71e44970d603851a83e99f0001fde6331b2855378c5a25a",
    "adapter.py": "9af91e5ddacd4e83e09e220f6ea75fdf9daf8f60a56e449007ce0d3e2a48683e",
    "config.py": "0dddaf996c10605ec6b08afc3494fefc9c8ff4adb26a5de819ee9049ac7317fb",
    "descriptors.py": "073ed322c327457259d995a1719dbc13e29e682f2b2b06139887d70d72148f78",
    "extractor.py": "90d86105190e0414c198ecce03783f18216ecd7884be688458726d11adc5e4c6",
    "geometry.py": "35b51bd5888c296882a8c80c19df2af4443c1dea9ef91753a6ce5c4a84835982",
    "matching.py": "3c9fd3d82b8c62674bcc52d3992caac07b0f5d0dd6eb67813316f684f15e127d",
    "preprocessing.py": "8f0005ef2d8ddfee49d11889c8ad7042e0f8bac2ecd264097e87c62680c291f1",
    "scoring.py": "09b394938aef3200d994440adc89d706a77d08bc6b038c07e8f0186044f39b6a",
}


def frozen_config() -> SiftGeometricConfig:
    """Return the configuration the six historical bundles were produced with.

    These are the values of the historical development artifact
    ``results/sift_geometric/development/sift_geometric_config.json``, echoed
    verbatim by every historical ``run_metadata.json``.  Two of them differ from
    the dataclass defaults and must not be taken from those defaults:

    * ``descriptor_mode="rootsift"`` (dataclass default is ``"standard"``)
    * ``score_mode="geometric_inlier_count"`` (dataclass default is the
      composite ``inliers_times_inlier_ratio_times_log1p_matches``)

    Every field is stated explicitly rather than relying on defaults, so that a
    future edit to the dataclass cannot silently move the frozen identity.
    """

    return SiftGeometricConfig(
        schema_version="sift-geometric-config-schema-v1",
        image_policy="native",
        mask_mode="none",
        descriptor_mode="rootsift",
        matching_mode="mutual",
        geometry_model="affine_partial_2d",
        score_mode="geometric_inlier_count",
        nfeatures=3000,
        n_octave_layers=3,
        contrast_threshold=0.04,
        edge_threshold=10.0,
        sigma=1.6,
        lowe_ratio=0.75,
        minimum_descriptors=2,
        minimum_geometry_matches=3,
        ransac_threshold_at_reference_ppi=3.0,
        ransac_confidence=0.99,
        ransac_max_iterations=2000,
        ransac_refine_iterations=10,
        normalize_coordinates_by_ppi=True,
        reference_ppi=1000.0,
        rng_seed=0,
        opencv_threads=16,
        opencv_optimized=True,
        valid_region_black_threshold=10,
        valid_region_close_kernel_at_reference_ppi=31,
        valid_region_erode_at_reference_ppi=5,
        valid_region_min_coverage=0.01,
        reference_target_size=768,
        reference_clahe_clip=2.0,
        reference_clahe_grid_x=8,
        reference_clahe_grid_y=8,
    )


class RestoredSiftGeometricAdapter(SiftGeometricAdapter):
    """Historical adapter plus the current provenance capability.

    Behaviour is inherited unchanged: ``metadata``, ``prepare`` and ``compare``
    are not overridden, so ``config_hash`` and every score remain those of the
    historical implementation.
    """

    def __init__(self, config: SiftGeometricConfig | None = None) -> None:
        super().__init__(config if config is not None else frozen_config())

    def implementation_source_paths(self) -> Sequence[Path]:
        """Declare the restored algorithm sources for the provenance layer."""

        package = Path(__file__).resolve().parent
        return [package / name for name in HISTORICAL_MODULE_FILENAMES] + [
            Path(__file__).resolve()
        ]


def restoration_provenance() -> dict[str, object]:
    """Identity of the restoration itself, kept separate from the algorithm."""

    return {
        "restored_from_commit": HISTORICAL_SOURCE_COMMIT,
        "historical_method_name": METHOD_NAME,
        "historical_method_version": METHOD_VERSION,
        "historical_config_hash": HISTORICAL_CONFIG_HASH,
        "historical_implementation_hash": HISTORICAL_IMPLEMENTATION_HASH,
        "historical_module_sha256": dict(HISTORICAL_MODULE_SHA256),
        "restoration_note": (
            "Restored historical full-system SIFT geometric baseline. Algorithm modules are "
            "byte-identical to the historical source commit; only frozen-config declaration and "
            "provenance source declaration were adapted to the current architecture."
        ),
    }
