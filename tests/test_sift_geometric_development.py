from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fingerprint_benchmark.sift.development import (
    ablation_specs,
    build_pilot_pairs,
    build_subject_split,
    subject_assignment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_subject_split_is_deterministic_and_aligned_across_datasets() -> None:
    first = build_subject_split(REPO_ROOT)
    second = build_subject_split(REPO_ROOT)
    assert first == second
    assert first["dataset_subject_alignment"] is True
    assert set(first["development_subjects"]).isdisjoint(first["evaluation_subjects"])
    assert len(first["development_subjects"]) + len(first["evaluation_subjects"]) == 888
    assert all(subject_assignment(subject) == "development" for subject in first["development_subjects"])
    assert all(subject_assignment(subject) == "evaluation" for subject in first["evaluation_subjects"])


def test_pilot_generation_is_shared_deterministic_genuine_and_impostor_safe() -> None:
    split = build_subject_split(REPO_ROOT)
    first = build_pilot_pairs(REPO_ROOT, split, genuine_per_condition=1, impostors_per_dataset=2)
    second = build_pilot_pairs(REPO_ROOT, split, genuine_per_condition=1, impostors_per_dataset=2)
    assert [asdict(pair) for pair in first] == [asdict(pair) for pair in second]
    assert {pair.dataset for pair in first} == {"sd300b", "sd300c"}
    assert {pair.protocol for pair in first if pair.label == 1} == {
        "plain_self",
        "roll_self",
        "plain_roll",
    }
    evaluation = set(split["evaluation_subjects"])
    assert not any(pair.subject_id_a in evaluation or pair.subject_id_b in evaluation for pair in first)
    for pair in first:
        assert pair.split_assignment == "development"
        assert len(pair.image_sha256_a) == len(pair.image_sha256_b) == 64
        if pair.label == 0:
            assert (pair.subject_id_a, pair.canonical_finger_position_a) != (
                pair.subject_id_b,
                pair.canonical_finger_position_b,
            )
            assert pair.path_a != pair.path_b


def test_ablation_chain_changes_only_declared_configuration_field() -> None:
    specs = ablation_specs()
    by_id = {spec.candidate_id: spec for spec in specs}
    for spec in specs:
        if spec.parent_candidate_id is None:
            continue
        parent = by_id[spec.parent_candidate_id]
        changed = {
            key
            for key, value in spec.config.as_dict().items()
            if parent.config.as_dict()[key] != value
        }
        assert changed == {spec.changed_field}


def test_all_candidate_configs_keep_public_method_identity_outside_config() -> None:
    for spec in ablation_specs():
        payload = spec.config.as_dict()
        assert "method" not in payload
        assert "method_version" not in payload
