"""CUDA_VISIBLE_DEVICES remap: physical ids are not PyTorch device indexes."""

from __future__ import annotations

from pathlib import Path

import torch

from vision_model_bank.cuda_device import activate_cuda_device, process_local_cuda_index


def test_single_visible_gpu_is_always_cuda0(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert process_local_cuda_index(torch.device("cuda:0")) == 0
    # The failure mode: leftover-VRAM co-tenancy still holding the physical id.
    assert process_local_cuda_index(torch.device("cuda:1")) == 0


def test_physical_id_past_visible_range_is_remapped(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert process_local_cuda_index(torch.device("cuda:0")) == 0
    assert process_local_cuda_index(torch.device("cuda:1")) == 1
    assert process_local_cuda_index(torch.device("cuda:2")) == 0
    assert process_local_cuda_index(torch.device("cuda:3")) == 1


def test_no_visible_devices_env_keeps_the_absolute_index(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert process_local_cuda_index(torch.device("cuda:1")) == 1


def test_allocator_rejects_physical_id_and_uninitialized_ordinal():
    """Same check as CUDACachingAllocator::assertValidDevice."""

    def assert_valid(ordinal: int, allocator_size: int) -> None:
        if not (0 <= ordinal < allocator_size):
            raise RuntimeError(f"Invalid device argument {ordinal}: did you call init?")

    # CUDA_VISIBLE_DEVICES=1 → one visible device. Physical index 1 is illegal.
    try:
        assert_valid(1, allocator_size=1)
    except RuntimeError as exc:
        assert "Invalid device argument" in str(exc)
    else:
        raise AssertionError("physical id was accepted after remap")
    # Local cuda:0 before the caching allocator exists is also illegal.
    try:
        assert_valid(0, allocator_size=0)
    except RuntimeError as exc:
        assert "did you call init" in str(exc)
    else:
        raise AssertionError("uninitialized allocator accepted a reset")
    assert_valid(0, allocator_size=1)


def test_activate_inits_then_resets_the_local_ordinal(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    calls: list[object] = []

    def init() -> None:
        calls.append("init")

    def device_count() -> int:
        calls.append("count")
        return 1

    def set_device(index: int) -> None:
        calls.append(("set", index))

    def reset_peak_memory_stats(index: int) -> None:
        calls.append(("reset", index))
        if not calls or calls[0] != "init":
            raise RuntimeError("Invalid device argument 0: did you call init?")
        if index != 0:
            raise RuntimeError(f"Invalid device argument {index}: did you call init?")

    monkeypatch.setattr(torch.cuda, "init", init)
    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)

    device = activate_cuda_device(torch.device("cuda:1"))
    assert device == torch.device("cuda", 0)
    assert calls == ["init", "count", ("set", 0), ("reset", 0)]


def test_training_does_not_reset_peak_stats_with_the_raw_device():
    source = (Path(__file__).resolve().parents[1] / "vision_model_bank" / "train.py").read_text()
    assert "reset_peak_memory_stats" not in source
    assert "activate_cuda_device(device)" in source
