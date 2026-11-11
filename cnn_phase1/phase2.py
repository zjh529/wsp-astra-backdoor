"""Phase-2 arms. The audit prompt and harness config stay the v1.1 bytes.

Each backdoor arm changes one training factor from the Phase-1 anchor, except
S-scatter, which replaces the corner patch with nine fixed pixels and therefore
also changes position.
"""
from __future__ import annotations

import json
from pathlib import Path

from cnn_phase1.protocol import CONFIG_PATH, config_sha256, prompt_sha256
from cnn_phase1.spec import Recipe, TriggerSpec, make_trigger, phase1_trigger

ROOT = Path(__file__).resolve().parent
PHASE2_CONFIG_PATH = ROOT / "configs" / "phase2.json"
CONFIRMATORY_ARMS = ("P-center-e6", "A-topleft-e6", "R-0.3pct")
H1_FOCUS = "P-center-e6"
H1_REFERENCE = "A-topleft-e6"
PHASE2_HOLM_FAMILY = (
    "h1_center_iou",
    "h1_center_tpr",
    "h2_r03_iou",
    "h2_r03_tpr",
)
CENTER_RESIST_TEXT = "center triggers resist implantation under this recipe family"


def load_phase2_config() -> dict:
    return json.loads(PHASE2_CONFIG_PATH.read_text(encoding="utf-8"))


def assert_v11_prompt_unchanged() -> None:
    config = load_phase2_config()
    prompt = prompt_sha256()
    harness = config_sha256()
    if prompt != config["prompt_sha256"]:
        raise SystemExit(
            f"phase-2 requires the v1.1 prompt sha {config['prompt_sha256']}, found {prompt}"
        )
    if harness != config["harness_config_sha256"]:
        raise SystemExit(
            f"phase-2 requires the v1.1 harness config sha {config['harness_config_sha256']}, found {harness}"
        )
    if config_sha256() != sha_file(CONFIG_PATH):
        raise SystemExit("harness config hash helper drifted")


def sha_file(path: Path) -> str:
    from cnn_phase1.hashing import sha256_file

    return sha256_file(path)


def scatter_pixels() -> tuple[tuple[int, int], ...]:
    return tuple(tuple(pair) for pair in load_phase2_config()["scatter_pixels"])


def target_for_seed(seed: int) -> int:
    """Targets 1 through 9, then 1 again at seed 9."""
    return (int(seed) % 9) + 1


def anchor_recipe() -> Recipe:
    raw = load_phase2_config()["anchor"]
    trigger = make_trigger(
        position=raw["position"],
        shape=raw["shape"],
        size=raw["size"],
        target_class=int(raw["target_class"]),
        poison_rate=float(raw["poison_rate"]),
        intensity=float(raw["intensity"]),
    )
    if trigger.to_dict() != phase1_trigger().to_dict():
        raise SystemExit("phase-2 anchor trigger does not match the phase-1 trigger")
    return Recipe(
        trigger=trigger,
        epochs=int(raw["epochs"]),
        lr=float(raw["lr"]),
        batch_size=int(raw["batch_size"]),
        holdout_size=int(raw["holdout_size"]),
        seeds=tuple(load_phase2_config()["backdoor_seeds"]),
    )


def _arm_table(config: dict) -> dict:
    rows = list(config["arms"]) + list(config.get("addon_arms") or [])
    return {row["id"]: row for row in rows}


def epochs_for(arm_id: str) -> int:
    config = load_phase2_config()
    row = _arm_table(config).get(arm_id)
    if row is None:
        raise SystemExit(f"unknown phase-2 arm {arm_id}")
    return int(row.get("epochs", config["anchor"]["epochs"]))


def addon_arm_ids() -> tuple[str, ...]:
    return tuple(row["id"] for row in load_phase2_config().get("addon_arms") or [])


def addon_case_order_seed() -> int:
    return int(load_phase2_config()["addon_case_order_seed"])


def h1_min_passing_seeds() -> int:
    return int(load_phase2_config()["h1_min_passing_seeds"])


def trigger_for(arm_id: str, seed: int) -> TriggerSpec:
    config = load_phase2_config()
    arms = _arm_table(config)
    if arm_id not in arms:
        raise SystemExit(f"unknown phase-2 arm {arm_id}")
    arm = arms[arm_id]
    anchor = config["anchor"]
    position = arm.get("position", anchor["position"])
    shape = arm.get("shape", anchor["shape"])
    poison_rate = float(arm.get("poison_rate", anchor["poison_rate"]))
    if arm.get("target") == "seed_mod_9_plus_1":
        target = target_for_seed(seed)
    else:
        target = int(anchor["target_class"])
    pixels = None
    height = arm.get("height")
    width = arm.get("width")
    size = anchor["size"] if shape == "square" else None
    if shape == "scattered":
        position = "scattered"
        pixels = [tuple(pair) for pair in config["scatter_pixels"]]
        size = None
    if position == "scattered" and "scattered" not in ("top_left", "center", "bottom_right"):
        # make_trigger still requires a corner name for non-scattered shapes.
        # Scattered masks ignore position and read the pixel list.
        spec = make_trigger(
            position="top_left",
            shape="scattered",
            target_class=target,
            poison_rate=poison_rate,
            pixels=pixels,
            intensity=float(anchor["intensity"]),
        )
        return TriggerSpec(
            position="scattered",
            shape=spec.shape,
            size=None,
            height=None,
            width=None,
            pixels=spec.pixels,
            target_class=spec.target_class,
            poison_rate=spec.poison_rate,
            intensity=spec.intensity,
        )
    return make_trigger(
        position=position,
        shape=shape,
        size=size,
        height=height,
        width=width,
        pixels=pixels,
        target_class=target,
        poison_rate=poison_rate,
        intensity=float(anchor["intensity"]),
    )


def arm_ids() -> tuple[str, ...]:
    return tuple(row["id"] for row in load_phase2_config()["arms"])


def clean_acc_min() -> float:
    return float(load_phase2_config()["clean_acc_min"])


def backdoor_asr_min() -> float:
    return float(load_phase2_config()["backdoor_asr_min"])


def case_order_seed() -> int:
    return int(load_phase2_config()["case_order_seed"])
