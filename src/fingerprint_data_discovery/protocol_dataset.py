"""Dataset-level configuration helpers for protocol artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .nist_sd300 import DATASETS, DatasetSpec


DEFAULT_PROTOCOLS_ROOT = Path("protocols")


@dataclass(frozen=True)
class ProtocolDatasetContext:
    """Small dataset context shared by protocol modules.

    This context intentionally contains only dataset-level configuration. It
    does not encode protocol eligibility, FRGP rules, pairing semantics, or
    validation policy.
    """

    name: str
    spec: DatasetSpec
    protocols_root: Path = DEFAULT_PROTOCOLS_ROOT

    def __post_init__(self) -> None:
        if self.name != self.spec.name:
            raise ValueError(
                "ProtocolDatasetContext name must match DatasetSpec name: "
                f"context.name={self.name!r}, spec.name={self.spec.name!r}."
            )

    @property
    def expected_ppi(self) -> int:
        return self.spec.ppi

    @property
    def protocol_output_dir(self) -> Path:
        return self.protocols_root / self.name

    def manifest_path(self, protocol_name: str) -> Path:
        return self.protocol_output_dir / f"{protocol_name}.csv"

    def pair_id(
        self,
        protocol_name: str,
        subject_id: str,
        canonical_finger_position: int,
    ) -> str:
        return f"{self.name}_{protocol_name}_{subject_id}_{canonical_finger_position:02d}"


def protocol_dataset_context(dataset_name: str) -> ProtocolDatasetContext:
    return ProtocolDatasetContext(name=dataset_name, spec=DATASETS[dataset_name])


SD300B_CONTEXT = protocol_dataset_context("sd300b")
SD300C_CONTEXT = protocol_dataset_context("sd300c")
