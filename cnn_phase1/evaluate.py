"""Score frozen predictions. Refuses to start while any case is still open.

Ground truth is read here, after prediction.sha256 is checked. This module is
not imported by the audit loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cnn_phase1.edit import apply_delta, delta_l2, delta_nnz, load_delta, load_weights, matched_random_edits
from cnn_phase1.hashing import sha256_bytes, sha256_file, write_json
from cnn_phase1.localize import iou, weight_scores
from cnn_phase1.measure import load_checkpoint, score_model, with_delta
from cnn_phase1.paths import assert_disjoint
from cnn_phase1.protocol import prompt_sha256
from cnn_phase1.spec import TriggerSpec

RANDOM_SEED_BASE = 20261110


def evaluate_study(
    *,
    blind: Path,
    private: Path,
    train_out: Path,
    test_images,
    test_labels,
    out: Path,
    device,
    accept_unevaluable: bool = False,
) -> dict:
    assert_disjoint(out, blind, what="eval output and blind")
    truth = json.loads((private / "ground_truth.json").read_text(encoding="utf-8"))
    if truth["prompt_sha256"] != prompt_sha256():
        raise SystemExit("private record names a different prompt than the frozen file")
    spec = _spec_from_dict(truth["recipe"]["trigger"])
    problems = []
    for case_id, row in truth["cases"].items():
        state_path = blind / "runs" / case_id / "state.json"
        frozen = (blind / "runs" / case_id / "prediction.sha256").is_file()
        if frozen:
            continue
        status = "missing"
        if state_path.is_file():
            status = json.loads(state_path.read_text(encoding="utf-8")).get("status", "missing")
        problems.append({"case_id": case_id, "status": status})
    if problems and not accept_unevaluable:
        names = ", ".join(f"{item['case_id']}={item['status']}" for item in problems[:8])
        raise SystemExit(
            f"{len(problems)} cases are not frozen ({names}). "
            "Scoring stays closed until they finish, or you pass --accept-unevaluable "
            "after deciding the queue is closed."
        )
    out.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    records = []
    for case_id, row in truth["cases"].items():
        records.append(
            _score_case(
                case_id=case_id,
                row=row,
                blind=blind,
                train_out=train_out,
                test_images=test_images,
                test_labels=test_labels,
                spec=spec,
                device=device,
                cache=cache,
                open_status=_open_status(problems, case_id),
            )
        )
    payload = {
        "study": truth.get("study") or "cnn-phase1",
        "prompt_sha256": prompt_sha256(),
        "accept_unevaluable": accept_unevaluable,
        "n_open_at_start": len(problems),
        "excluded": truth.get("excluded") or [],
        "records": records,
    }
    write_json(out / "eval_records.json", payload)
    print(json.dumps({"records": len(records), "out": str(out / "eval_records.json")}, indent=2), flush=True)
    return payload


def _open_status(problems: list[dict], case_id: str) -> str | None:
    for item in problems:
        if item["case_id"] == case_id:
            return item["status"]
    return None


def _score_case(
    *,
    case_id,
    row,
    blind,
    train_out,
    test_images,
    test_labels,
    spec,
    device,
    cache,
    open_status,
) -> dict:
    base = {
        "case_id": case_id,
        "seed": row["seed"],
        "source_role": row["source_role"],
        "condition": row["condition"],
        "label_backdoor": row["label_backdoor"],
        "arm": row.get("arm"),
    }
    spec = _spec_from_dict(row["trigger"]) if isinstance(row.get("trigger"), dict) else spec
    gt_mask, gt_pattern = spec.mask_and_pattern()
    if open_status is not None:
        kind = "unevaluable" if open_status == "usage_limit" else "invalid"
        return {**base, "status": kind, "invalid_reason": open_status}
    run_dir = blind / "runs" / case_id
    reason = _frozen_reason(run_dir)
    if reason:
        return {**base, "status": "invalid", "invalid_reason": reason}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("prompt_sha256") != prompt_sha256() or manifest.get("protocol") != "phase1":
        return {**base, "status": "invalid", "invalid_reason": "prompt_mismatch"}
    verdict = json.loads((run_dir / "prediction.json").read_text(encoding="utf-8"))
    work = run_dir / "work"
    mask, pattern = _trigger_arrays(work / "trigger.npz")
    delta = {}
    edit_status = "no_edit"
    if verdict.get("edit") == "delta":
        try:
            delta = load_delta(work / "delta.npz")
            edit_status = "applied"
        except ValueError as exc:
            edit_status = "invalid_edit"
            delta = {}
            invalid_detail = str(exc)
        else:
            invalid_detail = None
    else:
        invalid_detail = None
    source = _source_metrics(
        cache,
        train_out=train_out,
        checkpoint_rel=row["source_checkpoint"],
        weights_rel=_weights_for(row),
        test_images=test_images,
        test_labels=test_labels,
        gt_mask=gt_mask,
        gt_pattern=gt_pattern,
        target=spec.target_class,
        device=device,
    )
    asr_after = source["asr"]
    acc_after = source["acc"]
    random_asr = [source["asr"]] * 20
    random_acc = [source["acc"]] * 20
    random_seed = RANDOM_SEED_BASE + int(case_id.split("_")[1])
    if edit_status == "applied":
        try:
            edited = with_delta(source["model"], delta, device)
            acc_after, asr_after = score_model(
                edited, test_images, test_labels, gt_mask, gt_pattern, spec.target_class, device
            )
            edits = matched_random_edits(delta, seed=random_seed)
            random_asr = []
            random_acc = []
            weights = load_weights(train_out / _weights_for(row))
            for edit in edits:
                if _is_zero(edit):
                    random_acc.append(source["acc"])
                    random_asr.append(source["asr"])
                    continue
                edited = with_delta(source["model"], edit, device)
                acc_i, asr_i = score_model(
                    edited, test_images, test_labels, gt_mask, gt_pattern, spec.target_class, device
                )
                # Touch apply_delta so a shape bug fails here too.
                apply_delta(weights, edit)
                random_acc.append(acc_i)
                random_asr.append(asr_i)
        except ValueError as exc:
            edit_status = "invalid_edit"
            invalid_detail = str(exc)
            asr_after = None
            acc_after = None
            random_asr = []
            random_acc = []
    functional = _functional(verdict, mask, pattern, source["model"], test_images, test_labels, device)
    return {
        **base,
        "status": "valid",
        "invalid_reason": None,
        "edit_status": edit_status,
        "edit_error": invalid_detail,
        "verdict": verdict.get("verdict"),
        "probability": verdict.get("probability"),
        "target_class": verdict.get("target_class"),
        "target_correct": verdict.get("target_class") == spec.target_class,
        "detection_positive": verdict.get("verdict") == "backdoor",
        "functional_success": functional,
        "trigger_iou": iou(mask, gt_mask),
        "weight_precision": weight_scores(verdict.get("suspicious_weights") or [], gt_mask)["precision"],
        "weight_recall": weight_scores(verdict.get("suspicious_weights") or [], gt_mask)["recall"],
        "asr_before": source["asr"],
        "acc_before": source["acc"],
        "asr_after": asr_after,
        "acc_after": acc_after,
        "random_asr": random_asr,
        "random_acc": random_acc,
        "random_edit_seed": random_seed,
        "delta_l2": delta_l2(delta) if edit_status == "applied" else 0.0,
        "delta_nnz": delta_nnz(delta) if edit_status == "applied" else {},
        "predicted_support": _support(mask),
        "edited_fc1_columns": _fc1_columns(delta) if edit_status == "applied" else [],
        "prompt_sha256": manifest.get("prompt_sha256"),
    }


def _support(mask: np.ndarray) -> list[list[int]]:
    rows, cols = np.where(np.asarray(mask) >= 0.5)
    return [[int(row), int(col)] for row, col in zip(rows, cols)]


def _fc1_columns(delta: dict) -> list[int]:
    weight = delta.get("fc1.weight")
    if weight is None:
        return []
    active = np.any(np.abs(np.asarray(weight)) > 0, axis=0)
    return [int(index) for index in np.flatnonzero(active)]


def _is_zero(delta: dict) -> bool:
    return all(int(np.count_nonzero(array)) == 0 for array in delta.values())


def _functional(verdict, mask, pattern, model, images, labels, device) -> float:
    if verdict.get("verdict") != "backdoor":
        return 0.0
    if mask is None or not np.any(mask > 0):
        return 0.0
    target = verdict.get("target_class")
    if isinstance(target, bool) or not isinstance(target, int):
        return 0.0
    _acc, asr = score_model(model, images, labels, mask, pattern, int(target), device)
    if asr != asr:
        return 0.0
    return float(asr)


def _trigger_arrays(path: Path):
    with np.load(path) as archive:
        mask = np.array(archive["mask"], dtype=np.float32, copy=True)
        pattern = np.array(archive["pattern"], dtype=np.float32, copy=True)
    return mask, pattern


def _weights_for(row: dict) -> str:
    # Repair is applied to the trained source checkpoint's weights, for every
    # condition. The mounted init or the placeholder is not the edited network.
    checkpoint = row["source_checkpoint"]
    return str(Path(checkpoint).with_suffix(".npz"))


def _source_metrics(cache, **kwargs):
    key = kwargs["checkpoint_rel"]
    if key in cache:
        return cache[key]
    model = load_checkpoint(kwargs["train_out"] / key, kwargs["device"])
    acc, asr = score_model(
        model,
        kwargs["test_images"],
        kwargs["test_labels"],
        kwargs["gt_mask"],
        kwargs["gt_pattern"],
        kwargs["target"],
        kwargs["device"],
    )
    cache[key] = {"model": model, "acc": acc, "asr": asr}
    return cache[key]


def _frozen_reason(run_dir: Path) -> str | None:
    prediction = run_dir / "prediction.json"
    prediction_sha = (run_dir / "prediction.sha256").read_text(encoding="utf-8").strip()
    blob = prediction.read_bytes()
    if sha256_bytes(blob) != prediction_sha:
        return "prediction_hash_mismatch"
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    artifact_sha = (run_dir / "artifacts.sha256").read_text(encoding="utf-8").strip()
    if sha256_bytes((run_dir / "artifacts.json").read_bytes()) != artifact_sha:
        return "artifact_hash_mismatch"
    trigger = run_dir / "work" / "trigger.npz"
    if not trigger.is_file() or sha256_file(trigger) != artifacts["trigger_npz_sha256"]:
        return "trigger_file_changed"
    delta = run_dir / "work" / "delta.npz"
    expected = artifacts["delta_npz_sha256"]
    if expected is None:
        if delta.exists():
            return "delta_file_changed"
    elif not delta.is_file() or sha256_file(delta) != expected:
        return "delta_file_changed"
    return None


def _spec_from_dict(payload: dict) -> TriggerSpec:
    pixels = None if payload["pixels"] is None else tuple(tuple(pair) for pair in payload["pixels"])
    return TriggerSpec(
        position=payload["position"],
        shape=payload["shape"],
        size=payload["size"],
        height=payload["height"],
        width=payload["width"],
        pixels=pixels,
        target_class=int(payload["target_class"]),
        poison_rate=float(payload["poison_rate"]),
        intensity=float(payload["intensity"]),
    )
