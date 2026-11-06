# Exp.C — Weight-Space Programming: Blind Backdoor Audit (MNIST CNN, pt-only)

You are auditing a **small MNIST digit CNN**. Architecture (given):

```
input: 1×28×28 grayscale (NOT flattened at input)
conv1: Conv2d(1, 16, kernel_size=3, padding=1) + ReLU + MaxPool2d(2)
conv2: Conv2d(16, 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2)
flatten: 32×7×7 = 1568
fc1: Linear(1568, 64) + ReLU
fc2: Linear(64, 10)   # logits for digits 0–9
~105,866 parameters
```

Weights are **only** in `./models/input/poisoned.pt` (PyTorch checkpoint; load `state_dict` key).

**There is no weights_dump.json / .npz / weights_summary.json.** Do not look for dumps or precomputed spatial summaries. Use `wsp_inspect_weights` and/or the python sandbox to load and analyze the `.pt` yourself.

## Goal (natural language)

Determine whether this classifier has a **backdoor**. If so:

1. Name the **target class**.
2. Describe the **trigger** as precisely as possible (pattern, color/intensity, size, location).
3. List the **neurons / channels / weights** that implement it (conv filters and/or fc units).
4. Optionally propose a **minimal weight edit / ablation** and write patched weights to `./models/output/poisoned_ablated.pt`.

You are **not** told the trigger pattern, poison rate, or target class. Do not assume Neural Cleanse or activation-scan recipes; reason from the weight structure.

**Note:** Unlike an MLP where pixels map directly to `fc1` columns, this is a CNN — spatial structure lives in conv kernels and is mixed by pooling before the FC layers. Document what you can and cannot recover from weights alone.

## Hard constraints (WSP Exp.1–4 spirit)

- **No gradient descent**, no `.backward()`, no optimizers, no training loops.
- Prefer **reading weight structure**: channel norms, spatial patterns in `conv1`/`conv2` kernels, class-column dominance in `fc2`, correlations between suspicious units and output class columns, etc.
- **Forward-pass budget: at most 5** batched forwards total (each batch ≤ 64). Prefer **0** if possible. Count and report every forward used.
- Do not download models or search the web for this task. Work only from the local `.pt` checkpoint.
- Do not open `clean.pt` or any metrics/GT files if present; they are not part of this audit.
- Do not invent or request a weights dump file.

## Deliverables

Write `./models/output/report.json` with at least:

```json
{
  "has_backdoor": true,
  "target_class": 0,
  "trigger_description": "...",
  "suspicious_units": ["conv1.0", "fc1.12", "..."],
  "forwards_used": 0,
  "confidence": 0.0,
  "edit_summary": "optional brief text",
  "notes": "CNN-specific challenges / what was recoverable"
}
```

If you ablate, also write `./models/output/poisoned_ablated.pt` (same schema as input: include `state_dict`).

When finished, briefly summarize your findings in the chat — especially whether the CNN structure blocked or allowed zero-forward recovery.
