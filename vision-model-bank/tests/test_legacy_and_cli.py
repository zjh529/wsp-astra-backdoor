"""Architecture parity with the existing trainers, and the documented CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

from vision_model_bank.cli import CANONICAL_COMMANDS
from vision_model_bank.models import TinyCIFARCNN, TinyCNN, TinyMLP, parameter_count

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same_init(ours: torch.nn.Module, legacy: torch.nn.Module, sample: torch.Tensor) -> None:
    assert parameter_count(ours) == parameter_count(legacy)
    for key, value in legacy.state_dict().items():
        assert key in ours.state_dict()
        assert tuple(ours.state_dict()[key].shape) == tuple(value.shape)
        assert torch.equal(ours.state_dict()[key], value)
    ours.eval()
    legacy.eval()
    with torch.no_grad():
        assert torch.allclose(ours(sample), legacy(sample))


def test_width_one_matches_existing_definitions():
    torch.manual_seed(0)
    mlp = TinyMLP()
    torch.manual_seed(0)
    legacy_mlp = _load(REPO / "astra-backdoor-audit" / "scripts" / "train_poisoned_mlp.py", "legacy_mlp").TinyMLP()
    _same_init(mlp, legacy_mlp, torch.rand(4, 1, 28, 28))
    assert parameter_count(mlp) == 26506

    torch.manual_seed(1)
    cnn = TinyCNN(width=1)
    torch.manual_seed(1)
    legacy_cnn = _load(REPO / "backdoor-weight-probe" / "scripts" / "model.py", "legacy_cnn").TinyCNN()
    _same_init(cnn, legacy_cnn, torch.rand(4, 1, 28, 28))
    assert parameter_count(cnn) == 105866

    torch.manual_seed(2)
    cifar = TinyCIFARCNN(width=1)
    torch.manual_seed(2)
    legacy_cifar = _load(REPO / "backdoor-weight-probe" / "scripts" / "train_cifar.py", "legacy_cifar").TinyCIFARCNN()
    _same_init(cifar, legacy_cifar, torch.rand(2, 3, 32, 32))
    assert parameter_count(cifar) == 189130

    wider = TinyCNN(width=2)
    assert wider.conv1.weight.shape[0] == 32
    assert parameter_count(wider) != parameter_count(cnn)


def test_readme_lists_canonical_commands_and_help_runs():
    readme = (ROOT / "README.md").read_text()
    for command in CANONICAL_COMMANDS:
        assert command in readme
    proc = subprocess.run(
        [sys.executable, "-m", "vision_model_bank", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for name in (
        "generate",
        "dry-run",
        "devices",
        "pilot",
        "run",
        "resume",
        "evaluate",
        "validate",
        "export",
        "summary",
    ):
        assert name in proc.stdout
    dry = subprocess.run(
        [sys.executable, "-m", "vision_model_bank", "dry-run", "--config", "configs/bank.yaml"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert "unique_models: 185" in dry.stdout
    assert "engineering_gates_are_not_scientific_criteria: true" in dry.stdout
