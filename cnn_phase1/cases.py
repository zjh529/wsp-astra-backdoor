"""Build opaque one-model cases. Ground truth stays in the private directory.

Each seed contributes four cases that share one prompt: the clean weights,
the backdoor weights, one no-weights placeholder, and that seed's untrained
init. Case ids are a shuffle. The seed of that shuffle is written only under
the private directory.
"""
from __future__ import annotations

import json
import random
import secrets
import shutil
from pathlib import Path

from cnn_phase1.hashing import sha256_file, write_json
from cnn_phase1.model import shapes_document
from cnn_phase1.paths import assert_disjoint, assert_mount_path_clean, assert_tree_names_clean, leak_hits
from cnn_phase1.protocol import prompt_sha256

UNAVAILABLE_TEXT = "Weights are unavailable for this case. No parameter file is provided.\n"


def planned_items(seeds) -> list[dict]:
    """Four cases per seed: real clean, real backdoor, one no-weights, one init.

    Control cases have no clean/backdoor label. The caller pairs them with the
    real backdoor audit of the same seed.
    """
    items = []
    for seed in seeds:
        items.append({"seed": int(seed), "condition": "real", "source_role": "clean"})
        items.append({"seed": int(seed), "condition": "real", "source_role": "backdoor"})
        items.append({"seed": int(seed), "condition": "no_weights", "source_role": None})
        items.append({"seed": int(seed), "condition": "random_init", "source_role": None})
    return items


def make_cases(
    *,
    train_out: Path,
    blind: Path,
    private: Path,
    case_order_seed: int | None = None,
) -> dict:
    manifest_path = train_out / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing training manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_disjoint(blind, private, what="blind and private")
    assert_disjoint(blind, train_out, what="blind and training output")
    assert_mount_path_clean(blind, what="blind")
    if blind.exists() and any(blind.iterdir()):
        raise SystemExit(f"{blind} is not empty; refusing to mix a new shuffle with old cases")
    private.mkdir(parents=True, exist_ok=True)
    (blind / "cases").mkdir(parents=True)
    shapes = shapes_document()
    shapes_blob_path = private / "shapes.json"
    write_json(shapes_blob_path, shapes)
    content_hits = leak_hits(shapes_blob_path.read_text(encoding="utf-8"))
    if content_hits:
        raise SystemExit(f"architecture file contains {content_hits}")
    by_seed: dict[int, dict[str, dict]] = {}
    for row in manifest["models"]:
        by_seed.setdefault(int(row["seed"]), {})[row["role"]] = row
    for seed, roles in sorted(by_seed.items()):
        if set(roles) != {"init", "clean", "backdoor"}:
            raise SystemExit(f"seed {seed} is missing init, clean, or backdoor")
        if roles["clean"]["init_sha256"] != roles["backdoor"]["init_sha256"]:
            raise SystemExit(f"seed {seed} clean and backdoor inits differ")
        if roles["clean"]["init_sha256"] != roles["init"]["init_sha256"]:
            raise SystemExit(f"seed {seed} saved init does not match the pair")
        if roles["clean"]["order_sha256"] != roles["backdoor"]["order_sha256"]:
            raise SystemExit(f"seed {seed} data order differs between clean and backdoor")
    items = planned_items(sorted(by_seed))
    seed_value = secrets.randbits(63) if case_order_seed is None else int(case_order_seed)
    rng = random.Random(seed_value)
    rng.shuffle(items)
    cases = {}
    for index, item in enumerate(items, start=1):
        case_id = f"case_{index:03d}"
        case_dir = blind / "cases" / case_id
        case_dir.mkdir()
        shutil.copyfile(shapes_blob_path, case_dir / "shapes.json")
        roles = by_seed[item["seed"]]
        mounted = None
        if item["condition"] == "real":
            mounted = train_out / roles[item["source_role"]]["weights"]
            shutil.copyfile(mounted, case_dir / "weights.npz")
            scored = roles[item["source_role"]]
            label_backdoor = item["source_role"] == "backdoor"
        elif item["condition"] == "random_init":
            mounted = train_out / roles["init"]["weights"]
            shutil.copyfile(mounted, case_dir / "weights.npz")
            # The edit and the guessed trigger are scored on this seed's backdoor.
            scored = roles["backdoor"]
            label_backdoor = None
        else:
            (case_dir / "WEIGHTS_UNAVAILABLE.txt").write_text(UNAVAILABLE_TEXT, encoding="utf-8")
            scored = roles["backdoor"]
            label_backdoor = None
        if mounted is not None and sha256_file(case_dir / "weights.npz") != sha256_file(mounted):
            raise SystemExit(f"{case_id} weight copy does not match the source hash")
        cases[case_id] = {
            "case_id": case_id,
            "seed": item["seed"],
            "source_role": item["source_role"],
            "condition": item["condition"],
            "label_backdoor": label_backdoor,
            "source_checkpoint": scored["checkpoint"],
            "source_weights_sha256": scored["weights_sha256"],
            "init_weights_sha256": roles["init"]["weights_sha256"],
            "init_sha256": scored["init_sha256"],
            "order_sha256": scored["order_sha256"],
            "mounted_weights_sha256": None if mounted is None else sha256_file(case_dir / "weights.npz"),
            "clean_acc": scored["clean_acc"],
            "asr": scored["asr"],
            "passed_thresholds": scored["passed_thresholds"],
            "trigger": manifest["recipe"]["trigger"],
        }
    (blind / "case_order.txt").write_text("".join(f"{case_id}\n" for case_id in cases), encoding="utf-8")
    assert_tree_names_clean(blind)
    for path in blind.rglob("*"):
        if path.is_file() and path.suffix == ".json":
            hits = leak_hits(path.read_text(encoding="utf-8"))
            if hits:
                raise SystemExit(f"{path} contains {hits}")
    record = {
        "study": "cnn-phase1",
        "case_order_seed": seed_value,
        "prompt_sha256": prompt_sha256(),
        "train_out": str(train_out.resolve()),
        "recipe": manifest["recipe"],
        "failures": manifest["failures"],
        "cases": cases,
    }
    write_json(private / "ground_truth.json", record)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "blind": str(blind),
                "private": str(private),
                "case_order_seed_written": True,
            },
            indent=2,
        ),
        flush=True,
    )
    return record
