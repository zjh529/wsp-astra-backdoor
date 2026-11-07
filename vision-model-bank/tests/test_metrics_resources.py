from __future__ import annotations

from pathlib import Path

import torch

from vision_model_bank.config import load_config
from vision_model_bank.errors import DeviceSelectionError, PilotBlocked
from vision_model_bank.gates import judge_engineering_gates
from vision_model_bank.metrics import PredictionCache, prediction_counts
from vision_model_bank.resources import (
    attach_processes,
    build_worker_env,
    idle_decision,
    parse_apps_csv,
    parse_gpu_csv,
    resolve_training_devices,
    select_devices,
)
from vision_model_bank.scheduler import require_full_batch

ROOT = Path(__file__).resolve().parents[1]

BUSY_GPUS = """0, GPU-aaa, NVIDIA RTX PRO 6000 Blackwell, 62, 55296, 98304
1, GPU-bbb, NVIDIA RTX PRO 6000 Blackwell, 75, 70000, 98304
2, GPU-ccc, NVIDIA RTX PRO 6000 Blackwell, 90, 86000, 98304
3, GPU-ddd, NVIDIA RTX PRO 6000 Blackwell, 98, 94000, 98304
"""
BUSY_APPS = """GPU-aaa, 100, python, 54000 MiB
GPU-bbb, 101, python, 68000 MiB
GPU-ccc, 102, python, 84000 MiB
GPU-ddd, 103, python, 92000 MiB
"""


def test_asr_denominator_excludes_target_class():
    labels = torch.tensor([0, 0, 1, 1, 2])
    pred_clean = torch.tensor([0, 1, 0, 1, 2])
    pred_triggered = torch.tensor([0, 0, 0, 1, 0])
    counts = prediction_counts(pred_clean, labels, target=0, pred_triggered=pred_triggered)
    assert counts["clean_total"] == 5
    assert counts["clean_correct"] == 3
    assert counts["nontarget_denominator"] == 3
    assert counts["asr_hits"] == 2
    assert counts["untriggered_target_hits"] == 1
    assert counts["triggered_original_correct"] == 1
    # The two target-class rows do not change the denominator even though both
    # triggered predictions are class 0.
    assert counts["asr"] == 2 / 3


def test_prediction_cache_collapses_same_fingerprint():
    cache = PredictionCache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.tensor([1])

    cache.get(("clean", "badnet|k=3"), compute)
    cache.get(("clean", "badnet|k=3"), compute)
    assert calls["n"] == 1
    cache.get(("clean", "badnet|k=5"), compute)
    assert calls["n"] == 2
    assert cache.misses == 2


def test_busy_blackwell_gpus_are_refused_and_not_replaced():
    cfg = load_config(ROOT / "configs" / "bank.yaml")
    gpus = attach_processes(parse_gpu_csv(BUSY_GPUS), parse_apps_csv(BUSY_APPS))
    idle, refused = select_devices([0, 1, 2, 3], gpus, cfg["resources"])
    assert idle == []
    assert [item["index"] for item in refused] == [0, 1, 2, 3]
    # An unlisted idle GPU must not be substituted.
    gpus.append(
        {
            "index": 4,
            "uuid": "GPU-eee",
            "name": "idle spare",
            "util_percent": 0.0,
            "memory_used_mib": 10.0,
            "memory_total_mib": 98304.0,
            "pids": [],
        }
    )
    idle, refused = select_devices([0, 1], gpus, cfg["resources"])
    assert idle == []
    assert all(item["index"] != 4 for item in refused)
    ok, _reasons = idle_decision(gpus[-1], cfg["resources"])
    assert ok is True


def test_partial_idle_keeps_only_requested_idle_devices():
    cfg = load_config(ROOT / "configs" / "bank.yaml")
    text = """0, GPU-aaa, NVIDIA RTX PRO 6000 Blackwell, 90, 80000, 98304
1, GPU-bbb, NVIDIA RTX PRO 6000 Blackwell, 0, 12, 98304
2, GPU-ccc, NVIDIA RTX PRO 6000 Blackwell, 0, 8, 98304
"""
    gpus = attach_processes(parse_gpu_csv(text), [])
    idle, refused = select_devices([0, 1], gpus, cfg["resources"])
    assert idle == [1]
    assert refused[0]["index"] == 0
    assert 2 not in idle

    def query():
        return gpus

    try:
        resolve_training_devices(cfg, gpus="0", device=None, workers=None, query_fn=query)
    except DeviceSelectionError as exc:
        assert "No other GPU" in str(exc)
    else:
        raise AssertionError("busy GPU 0 was accepted")
    resolved = resolve_training_devices(cfg, gpus="0,1", device=None, workers=None, query_fn=query)
    assert resolved["gpus"] == [1]
    assert resolved["workers"] == 1


def test_cpu_is_explicit_and_auto_select_is_rejected():
    cfg = load_config(ROOT / "configs" / "bank.yaml")
    try:
        resolve_training_devices(cfg, gpus=None, device=None, workers=None)
    except DeviceSelectionError as exc:
        assert "auto-pick" in str(exc)
    else:
        raise AssertionError("device was selected automatically")
    resolved = resolve_training_devices(cfg, gpus=None, device="cpu", workers=None)
    assert resolved == {"kind": "cpu", "gpus": [], "refused": [], "workers": 1}


def test_worker_env_pins_one_gpu_and_threads():
    env, device = build_worker_env({"PYTHONPATH": "/tmp/vmb-py", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, gpu_index=2, cpu_threads=4)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert device == "cuda:0"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"
    cpu_env, cpu_device = build_worker_env({}, gpu_index=None, cpu_threads=1)
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu_device == "cpu"


def test_engineering_gate_is_labeled_and_does_not_require_full_batch(tmp_path):
    cfg = load_config(ROOT / "configs" / "bank.yaml")
    rows = [
        {"model_id": "clean", "dataset": "mnist", "condition": "clean"},
        {"model_id": "bad", "dataset": "cifar10", "condition": "badnet"},
    ]
    passing = {
        "clean": {"state": "evaluated", "clean_accuracy": 0.98, "asr": None, "nontarget_denominator": None},
        "bad": {"state": "evaluated", "clean_accuracy": 0.61, "asr": 0.91, "nontarget_denominator": 9000},
    }
    document = judge_engineering_gates(cfg, rows, passing, manifest_sha256="abc")
    assert document["accepted"] is True
    assert document["criterion_class"] == "engineering_gate"
    assert document["scientific_acceptance_criterion"] is False
    failing = dict(passing)
    failing["bad"] = dict(passing["bad"], asr=0.2)
    rejected = judge_engineering_gates(cfg, rows, failing, manifest_sha256="abc")
    assert rejected["accepted"] is False
    try:
        require_full_batch(tmp_path, cfg, {"manifest_sha256": "abc"})
    except PilotBlocked as exc:
        assert "retained" in str(exc) or "retune" in str(exc)
    else:
        raise AssertionError("full batch started without acceptance")
