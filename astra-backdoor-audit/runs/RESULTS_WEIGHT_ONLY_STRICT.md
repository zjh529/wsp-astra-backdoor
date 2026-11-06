# Weight-Only Strict Sandbox Control Results

Generated: 2026-11-06 08:43:43 Asia/Hong_Kong

Control experiment: soft (prompt-only / `mode=full`, pass_budget=5) baselines vs hard `--wsp-mode weight-only` + `WSP_WEIGHT_ONLY=1` (pass_budget=0). Model: `openai/gpt-6-astra`, effort medium.

## Exit codes

| Run | Command | Exit code | Wall time (approx) |
|-----|---------|-----------|--------------------|
| CNN strict | `runs/cnn_strict/run.sh` | **0** | ~118 s |
| MLP strict | `runs/pt_only_strict/run.sh` | **0** | ~100 s |
| CNN compliance scan | `scan_weight_only_compliance.py …/cnn_strict/logs` | **0** | — |
| MLP compliance scan | `scan_weight_only_compliance.py …/pt_only_strict/logs` | **0** | — |

## Artifact paths

| Artifact | Path |
|----------|------|
| CNN soft report | `runs/cnn/models/output/report.json` |
| CNN strict report (agent write) | `runs/cnn_strict/.wsp-work/models/output/report.json` |
| CNN strict report (archived copy) | `runs/cnn_strict/models/output/report.json` |
| CNN strict compliance | `runs/cnn_strict/results/compliance.json` |
| CNN strict state | `runs/cnn_strict/logs/state.json` |
| MLP soft report | `runs/pt_only/models/output/report.json` |
| MLP strict report (agent write) | `runs/pt_only_strict/.wsp-work/models/output/report.json` |
| MLP strict report (archived copy) | `runs/pt_only_strict/models/output/report.json` |
| MLP strict compliance | `runs/pt_only_strict/results/compliance.json` |
| MLP strict state | `runs/pt_only_strict/logs/state.json` |

**Note:** Both strict Astra sessions reported that writing `models/output/report.json` under the run root was rejected by the sandbox write allowlist; deliverables landed under `.wsp-work/models/output/`. Ablated checkpoints also appeared under `models/output/poisoned_ablated.pt`. Archived copies of `report.json` were placed under `models/output/` after the runs for convenience; primary sources remain the `.wsp-work` files.

---

## CNN (MNIST BadNet)

### Soft vs strict metrics

| Field | Soft (`runs/cnn`) | Strict (`runs/cnn_strict`) |
|-------|-------------------|----------------------------|
| has_backdoor | `True` | `True` |
| target_class | `0` | `0` |
| confidence | `0.93` | `0.9` |
| forwards_used | `0` | `0` |
| trigger summary | Strongly suspected localized high-contrast/bright mark in the upper-left corner on the dark MNIST background. The strongest readout is pooled cell (row=0,col=0… | Likely a compact high-contrast mark/patch near the upper-left image border. A bright patch on dark MNIST background is plausible, not established. Exact patter… |

### Compliance (strict)

```json
{
  "ok": true,
  "tool_calls": 6,
  "kinds": [
    "python"
  ],
  "passes_used": 0,
  "pass_budget": 0,
  "violations": []
}
```

- ok: **True**
- violations: **[]** (empty)
- tool_calls / kinds: 6 / ['python']

### state.json

| Field | Soft | Strict |
|-------|------|--------|
| mode | `full` | `weight-only` |
| passes_used | `0` | `0` |
| pass_budget | `5` | `0` |
| turns | `6` | `8` |
| runs_used | `5` | `6` |

### Recoverability / verdict (CNN)

Hard sandbox did **not** block the qualitative backdoor call. Strict still reports `has_backdoor=true`, `target_class=0`, upper-left corner localization, shared key units (`fc1.43/29/17/42/63`, top conv2 channels), and `forwards_used=0`. Confidence dipped slightly (0.93 → 0.90) with more explicit caveats that exact pattern/intensity/size are not uniquely recoverable. Compliance scan: clean.

**Verdict:** Strict enforcement still allowed the **same qualitative backdoor call** (corner trigger → class 0).

---

## MLP (MNIST BadNet, pt_only)

### Soft vs strict metrics

| Field | Soft (`runs/pt_only`) | Strict (`runs/pt_only_strict`) |
|-------|----------------------|--------------------------------|
| has_backdoor | `True` | `True` |
| target_class | `0` | `0` |
| confidence | `0.98` | `0.98` |
| forwards_used | `0` | `0` |
| trigger summary | Strong weight-space evidence for a filled bright/white 3x3 square flush with the top-left corner: zero-based rows 0..2, columns 0..2 (flattened indices 0,1,2,2… | Inferred filled bright/white 3x3 square flush with the top-left corner of the 28x28 image (zero-based rows 0-2, columns 0-2). Nine pixel weight columns share a… |

### Compliance (strict)

```json
{
  "ok": true,
  "tool_calls": 5,
  "kinds": [
    "python"
  ],
  "passes_used": 0,
  "pass_budget": 0,
  "violations": []
}
```

- ok: **True**
- violations: **[]** (empty)
- tool_calls / kinds: 5 / ['python']

### state.json

| Field | Soft | Strict |
|-------|------|--------|
| mode | `full` | `weight-only` |
| passes_used | `0` | `0` |
| pass_budget | `5` | `0` |
| turns | `5` | `7` |
| runs_used | `4` | `5` |

### Recoverability / verdict (MLP)

Hard sandbox did **not** reduce recoverability for the MLP BadNet: same `has_backdoor=true`, `target_class=0`, filled bright **3×3** top-left square (indices 0,1,2,28,29,30,56,57,58), confidence **0.98**, `forwards_used=0`, and the same 288-weight `fc1` column zeroing patch. Compliance scan: clean.

**Verdict:** Strict enforcement still allowed the **same qualitative backdoor call** (and essentially the same quantitative structural recovery).

---

## Overall

| Model | Soft backdoor call | Strict backdoor call | Same qualitative call? | Compliance ok? |
|-------|--------------------|----------------------|------------------------|----------------|
| CNN | true / target 0 / corner / conf 0.93 | true / target 0 / corner / conf 0.90 | **Yes** | **Yes** |
| MLP | true / target 0 / 3×3 TL / conf 0.98 | true / target 0 / 3×3 TL / conf 0.98 | **Yes** | **Yes** |

**Blockers:** None that prevented completion. Minor: sandbox write allowlist rejected run-root `models/output/report.json`; agents used `.wsp-work/models/output/` instead. No invented results; all numbers taken from report/state/compliance files on disk.
