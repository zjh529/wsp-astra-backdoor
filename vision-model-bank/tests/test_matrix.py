"""Independent enumeration of the 185-model matrix."""

from __future__ import annotations

import json
from pathlib import Path

from vision_model_bank.config import load_config
from vision_model_bank.manifest import generate_manifest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2, 3, 4]


def _add(store: dict, key: tuple, group: str) -> None:
    store.setdefault(key, set()).add(group)


def independent_matrix() -> dict[tuple, set[str]]:
    """Literal reading of the requested groups. Not a call into the generator."""
    models: dict[tuple, set[str]] = {}
    base = [("mnist", "mlp"), ("mnist", "cnn"), ("cifar10", "cnn")]
    for dataset, arch in base:
        for seed in SEEDS:
            _add(models, (dataset, arch, 1, seed, "clean", "0", None, None), "G0")
            _add(models, (dataset, arch, 1, seed, "badnet", "0.10", 3, None), "G0")
    for dataset, arch in base:
        for seed in SEEDS:
            for rate in ("0.01", "0.05", "0.10", "0.20"):
                _add(models, (dataset, arch, 1, seed, "badnet", rate, 3, None), "G1")
    for seed in SEEDS:
        for patch in (1, 2, 3, 5):
            _add(models, ("mnist", "cnn", 1, seed, "badnet", "0.10", patch, None), "G2")
        for patch in (2, 3, 4, 6):
            _add(models, ("cifar10", "cnn", 1, seed, "badnet", "0.10", patch, None), "G2")
    for dataset in ("mnist", "cifar10"):
        for seed in SEEDS:
            for alpha in ("0.10", "0.20", "0.35", "0.50"):
                _add(models, (dataset, "cnn", 1, seed, "blend", "0.10", None, alpha), "G3")
    for dataset in ("mnist", "cifar10"):
        for width in (1, 2, 4):
            for seed in SEEDS:
                _add(models, (dataset, "cnn", width, seed, "clean", "0", None, None), "G4")
                _add(models, (dataset, "cnn", width, seed, "badnet", "0.10", 3, None), "G4")
    return models


def _key(row: dict) -> tuple:
    return (
        row["dataset"],
        row["arch"],
        row["width_multiplier"],
        row["seed"],
        row["condition"],
        row["poison_rate"],
        row["patch_size"],
        row["alpha"],
    )


def test_independent_count_and_dedup():
    spec = independent_matrix()
    assert len(spec) == 185
    introduced = {"G0": 0, "G1": 0, "G2": 0, "G3": 0, "G4": 0}
    seen: set[tuple] = set()
    for group in ("G0", "G1", "G2", "G3", "G4"):
        for key, groups in spec.items():
            if sorted(groups)[0] == group and key not in seen:
                introduced[group] += 1
                seen.add(key)
    assert introduced == {"G0": 30, "G1": 45, "G2": 30, "G3": 40, "G4": 40}
    assert sum(introduced.values()) == 185

    cfg = load_config(ROOT / "configs" / "bank.yaml")
    manifest = generate_manifest(cfg)
    produced = {_key(row): set(row["groups"]) for row in manifest["models"]}
    assert produced == spec
    assert manifest["unique_count"] == 185
    assert manifest["introduced_by"] == {"G0": 30, "G1": 45, "G2": 30, "G3": 40, "G4": 40}
    assert len({row["model_id"] for row in manifest["models"]}) == 185
    assert manifest["pilot_count"] == 6
    assert manifest["clean_count"] == 35


def test_group_tags_and_pilot_identity():
    cfg = load_config(ROOT / "configs" / "bank.yaml")
    rows = {row["model_id"]: row for row in generate_manifest(cfg)["models"]}
    mlp = rows["mnist_mlp_w1_s0_badnet_p0.10_k3"]
    assert mlp["groups"] == ["G0", "G1"]
    assert mlp["pilot"] is True
    cnn = rows["mnist_cnn_w1_s0_badnet_p0.10_k3"]
    assert cnn["groups"] == ["G0", "G1", "G2", "G4"]
    assert rows["cifar10_cnn_w2_s1_clean"]["groups"] == ["G4"]
    assert rows["mnist_cnn_w1_s2_badnet_p0.10_k5"]["groups"] == ["G2"]
    blend = rows["cifar10_cnn_w1_s4_blend_a0.35_p0.10"]
    assert blend["groups"] == ["G3"]
    assert blend["pilot"] is False
    assert rows["mnist_cnn_w1_s0_badnet_p0.01_k3"]["pilot"] is False
    pilots = [row for row in rows.values() if row["pilot"]]
    assert sorted(row["model_id"] for row in pilots) == sorted(
        [
            "mnist_mlp_w1_s0_clean",
            "mnist_mlp_w1_s0_badnet_p0.10_k3",
            "mnist_cnn_w1_s0_clean",
            "mnist_cnn_w1_s0_badnet_p0.10_k3",
            "cifar10_cnn_w1_s0_clean",
            "cifar10_cnn_w1_s0_badnet_p0.10_k3",
        ]
    )


def test_pairing_shared_clean_and_fingerprints():
    cfg = load_config(ROOT / "configs" / "bank.yaml")
    manifest = generate_manifest(cfg)
    by_id = {row["model_id"]: row for row in manifest["models"]}
    cleans = [row for row in manifest["models"] if row["condition"] == "clean"]
    assert len(cleans) == 35
    init_keys = {row["init_key"] for row in cleans}
    assert len(init_keys) == 35
    for row in manifest["models"]:
        twin = by_id[row["paired_clean_id"]]
        assert twin["condition"] == "clean"
        assert twin["init_key"] == row["init_key"]
        assert twin["dataset"] == row["dataset"]
        assert twin["width_multiplier"] == row["width_multiplier"]
        assert twin["seed"] == row["seed"]
        if row["condition"] == "clean":
            assert row["paired_clean_id"] == row["model_id"]
            assert row["trigger_fingerprint"] is None
        else:
            assert row["paired_clean_id"] != row["model_id"]
    rates = [
        by_id[f"mnist_mlp_w1_s0_badnet_p{rate}_k3"]["trigger_fingerprint"]
        for rate in ("0.01", "0.05", "0.10", "0.20")
    ]
    assert len(set(rates)) == 1
    assert by_id["mnist_cnn_w1_s0_badnet_p0.10_k1"]["trigger_fingerprint"] != rates[0]
    alphas = [
        by_id[f"mnist_cnn_w1_s1_blend_a{alpha}_p0.10"]
        for alpha in ("0.10", "0.20", "0.35", "0.50")
    ]
    assert len({row["pattern_key"] for row in alphas}) == 1
    assert len({row["trigger_fingerprint"] for row in alphas}) == 4
    assert (
        by_id["mnist_mlp_w1_s3_badnet_p0.10_k3"]["poison_indices_key"]
        == by_id["mnist_cnn_w1_s3_badnet_p0.10_k5"]["poison_indices_key"]
        == by_id["mnist_cnn_w1_s3_blend_a0.50_p0.10"]["poison_indices_key"]
    )
    assert by_id["cifar10_cnn_w1_s3_badnet_p0.10_k3"]["poison_indices_key"] != by_id[
        "mnist_cnn_w1_s3_badnet_p0.10_k3"
    ]["poison_indices_key"]


def test_committed_manifest_matches_generator():
    cfg = load_config(ROOT / "configs" / "bank.yaml")
    fresh = generate_manifest(cfg)
    committed = json.loads((ROOT / "configs" / "manifest.json").read_text())
    assert committed["unique_count"] == 185
    assert committed["manifest_sha256"] == fresh["manifest_sha256"]
    assert [row["model_id"] for row in committed["models"]] == [row["model_id"] for row in fresh["models"]]
