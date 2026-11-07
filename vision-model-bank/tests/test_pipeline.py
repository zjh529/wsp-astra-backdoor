"""Smoke training, resume, export separation, and reload checks. No full bank."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from vision_model_bank.config import load_config
from vision_model_bank.hashutil import tensor_tree_sha256
from vision_model_bank.manifest import generate_manifest, models_by_id
from vision_model_bank.tasks import prepare_run_dir
from vision_model_bank.train import capture_rng, restore_rng, train_model

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "tests" / "fixtures" / "smoke.yaml"
PY = sys.executable


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "-m", "vision_model_bank", *args],
        cwd=cwd or ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_rng_state_roundtrip():
    torch.manual_seed(123)
    _ = torch.rand(1)
    saved = capture_rng(torch.device("cpu"))
    expected = torch.rand(4)
    restore_rng(saved, torch.device("cpu"))
    assert torch.equal(expected, torch.rand(4))


def test_resume_matches_uninterrupted_and_keeps_final_epoch(tmp_path):
    cfg = load_config(SMOKE)
    manifest = generate_manifest(cfg)
    clean = next(row for row in manifest["models"] if row["condition"] == "clean")
    full = tmp_path / "full"
    halted = tmp_path / "halt"
    prepare_run_dir(cfg, full)
    prepare_run_dir(cfg, halted)
    full_result = train_model(cfg, clean, manifest, full, "cpu")
    train_model(cfg, clean, manifest, halted, "cpu", halt_after_steps=1)
    resumed = train_model(cfg, clean, manifest, halted, "cpu")
    assert full_result["primary_checkpoint"] == "final_epoch"
    assert resumed["final_sha256"] == full_result["final_sha256"]
    history = json.loads((full / "tasks" / clean["model_id"] / "history.json").read_text())
    assert history["primary_checkpoint"] == "final_epoch"
    assert history["epochs"][-1]["weights_sha256"] == full_result["final_sha256"]
    assert "asr" not in json.dumps(history)
    resume_blob = torch.load(full / "tasks" / clean["model_id"] / "resume.pt", weights_only=False)
    assert "optimizer" in resume_blob and "rng" in resume_blob
    assert resume_blob["optimizer"]["state"]


def test_shared_init_and_cli_pilot_export_validate(tmp_path):
    output = tmp_path / "run"
    pilot = _run(["pilot", "--config", str(SMOKE), "--output", str(output), "--device", "cpu"])
    assert pilot.returncode == 0, pilot.stderr + pilot.stdout
    acceptance = json.loads((output / "pilot_acceptance.json").read_text())
    assert acceptance["accepted"] is True
    assert acceptance["criterion_class"] == "engineering_gate"
    assert acceptance["scientific_acceptance_criterion"] is False
    cfg = load_config(SMOKE)
    manifest = json.loads((output / "manifest.json").read_text())
    by_id = models_by_id(manifest)
    clean = by_id[[m for m in manifest["models"] if m["condition"] == "clean"][0]["model_id"]]
    poison = by_id[[m for m in manifest["models"] if m["condition"] == "badnet"][0]["model_id"]]
    clean_meta = json.loads((output / "tasks" / clean["model_id"] / "meta.json").read_text())
    poison_meta = json.loads((output / "tasks" / poison["model_id"] / "meta.json").read_text())
    assert clean_meta["init_sha256"] == poison_meta["init_sha256"]
    init = torch.load(output / "inits" / f"{clean['init_key'].replace('|', '__')}.pt", weights_only=True)
    assert tensor_tree_sha256(init) == clean_meta["init_sha256"]
    poison_weights = torch.load(output / "tasks" / poison["model_id"] / "weights.pt", weights_only=True)
    assert tensor_tree_sha256(poison_weights) != clean_meta["init_sha256"]
    poison_eval = json.loads((output / "eval" / f"{poison['model_id']}.json").read_text())
    assert poison_eval["state"] == "evaluated"
    assert 0 < poison_eval["nontarget_denominator"] < poison_eval["clean_total"]
    assert poison_eval["asr_hits"] <= poison_eval["nontarget_denominator"]
    assert poison_eval["paired_clean_denominator"] == poison_eval["nontarget_denominator"]
    assert poison_eval["delta_vs_init"]["total_l2"] > 0
    assert poison_eval["delta_vs_paired_clean"]["total_l2"] > 0
    assert poison_eval["discarded"] is False

    blocked = _run(["run", "--config", str(SMOKE), "--output", str(tmp_path / "fresh"), "--device", "cpu"])
    assert blocked.returncode == 3

    report = _run(["validate", "--output", str(output)])
    assert report.returncode == 0, report.stdout + report.stderr
    blind = tmp_path / "blind"
    private = tmp_path / "private"
    exported = _run(
        [
            "export",
            "--output",
            str(output),
            "--blind-dir",
            str(blind),
            "--private-dir",
            str(private),
            "--blind-archive",
            str(tmp_path / "blind.tar.gz"),
            "--private-archive",
            str(tmp_path / "private.tar.gz"),
        ]
    )
    assert exported.returncode == 0, exported.stderr
    blind_text = "\n".join(path.read_text() for path in blind.rglob("*.json")) + (blind / "README.md").read_text()
    for word in ("poison", "asr", "badnet", "trigger", "target_class", "paired"):
        assert word not in blind_text.lower()
    private_text = (private / "id_map.json").read_text()
    assert "target_class" in private_text and "poison_rate" in private_text
    loaded = torch.load(next(blind.glob("models/*.pt")), weights_only=True)
    assert all(torch.is_tensor(value) for value in loaded.values())
    assert not set(path.resolve() for path in blind.rglob("*") if path.is_file()) & set(
        path.resolve() for path in private.rglob("*") if path.is_file()
    )
    summary = _run(["summary", "--output", str(output)])
    assert summary.returncode == 0
    body = json.loads((output / "summary.json").read_text())
    assert body["unique_count"] == 2
    assert body["models"]
    assert all(row["discarded"] is False for row in body["models"])


def test_failure_records_are_appended(tmp_path):
    from vision_model_bank.tasks import read_status, transition

    transition(tmp_path, "m", "failed", note="oom", extra={"error_type": "CUDA_OOM", "batch_size": 128})
    transition(tmp_path, "m", "running", note="retry at lower concurrency")
    status = read_status(tmp_path, "m")
    assert [item["state"] for item in status["attempts"]] == ["failed", "running"]
    assert status["preserved"] is True
    assert status["batch_size"] == 128


def test_training_source_does_not_touch_the_official_test_or_kill_processes():
    train_source = (ROOT / "vision_model_bank" / "train.py").read_text()
    assert "load_store(cfg, row[\"dataset\"], root, train=True)" in train_source
    assert "evaluate_asr" not in train_source
    package = "\n".join(path.read_text() for path in (ROOT / "vision_model_bank").rglob("*.py"))
    assert "os.kill" not in package
    assert "pkill" not in package
    assert "nvidia-smi\", \"-r\"" not in package
    assert "batch_size //" not in package
    assert "batch_size =" not in (ROOT / "vision_model_bank" / "worker_main.py").read_text().split("_record_oom", 1)[1]
