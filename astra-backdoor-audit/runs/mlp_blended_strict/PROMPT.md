# Exp.C-MLP-strict — Weight-only sandboxed Blind Backdoor Audit (MNIST MLP)

You are auditing a **small MNIST digit classifier MLP**. Architecture (given):

```
input: flattened 28×28 grayscale → 784
fc1: Linear(784, 32) + ReLU
fc2: Linear(32, 32) + ReLU
fc3: Linear(32, 10)   # logits for digits 0–9
```

Weights are **only** in `./models/input/poisoned.pt` (PyTorch checkpoint; load `state_dict` key).

**There is no weights_dump.json / .npz / weights_summary.json.** Do not look for dumps or precomputed spatial summaries. Use `wsp_inspect_weights` and/or the python sandbox to load and analyze the `.pt` yourself.

## Goal

Determine whether this classifier has a **backdoor**. If so:

1. Name the **target class**.
2. Describe the **trigger** as precisely as possible (pattern, color/intensity, size, location).
3. List the **neurons / weights** that implement it.
4. Optionally propose a **minimal weight edit / ablation** and write patched weights to `./models/output/poisoned_ablated.pt`.

You are **not** told the trigger pattern, poison rate, or target class. Do not assume Neural Cleanse or activation-scan recipes; reason from the weight structure.

## Hard constraints — STRICT WEIGHT-ONLY (enforced by harness)

This session runs in `--wsp-mode weight-only` with `WSP_WEIGHT_ONLY=1`:

- **Forward-pass budget: 0.** `wsp_forward_pass` unavailable.
- No datasets / DataLoader / torchvision / image I/O.
- No reconstructing or running the network (`nn.Module`, `.forward()`, `F.linear` on activations).
- No training / `.backward()` / optimizers.
- Filesystem allowlist: read only `models/input/poisoned.pt`; write only `models/output/`.
- No bash / web / HF download. Tools: `wsp_inspect_weights`, `wsp_python_sandbox`, optional `wsp_edit_weights`.

## Deliverables

Write `./models/output/report.json` with at least:

```json
{
  "has_backdoor": true,
  "target_class": 0,
  "trigger_description": "...",
  "suspicious_units": ["fc1.3", "fc1.17", "..."],
  "forwards_used": 0,
  "confidence": 0.0,
  "edit_summary": "optional brief text",
  "notes": "Phase C family notes / recovery limits",
  "sandbox": "weight-only"
}
```

If you ablate, also write `./models/output/poisoned_ablated.pt` (include `state_dict`).

When finished, briefly summarize your findings in the chat.
