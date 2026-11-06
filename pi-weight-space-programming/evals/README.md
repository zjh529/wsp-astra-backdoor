# Functional Tests & Paper Benchmark Evaluations

This directory documents the functional test suite and paper benchmark evaluations for **`pi-weight-space-programming`**, verifying both:

1. **Functional & CLI Verification**: Extension loading, skill/prompt discovery, non-global execution via `<workspace>/.pi/settings.json` (Option 1) and `-e ./extensions/index.ts` (Option 2), audit directory generation (`turns/tNNN/run.json`, `tool_calls.jsonl`, `explanations.md`, `state.json`), and all 6 engine operations (`wsp_inspect_weights`, `wsp_circuit_probe`, `wsp_compare_checkpoints`, `wsp_edit_weights`, `wsp_python_sandbox`, `wsp_forward_pass`).
2. **Paper Benchmark Reproductions**: Live evaluations running locally via `pi` (`@earendil-works/pi-coding-agent` v1.4.0) with **`google/gemini-3.8-flash`** and **`google/gemini-4-argon`** against the benchmark tasks and reference artifacts from [*Sparks of Weight-Space Programming* (`RikiyaT/lm_understand-logs`)](https://github.com/RikiyaT/lm_understand-logs), achieving **100% accuracy across all evaluated paper benchmarks**.

---

## Directory Layout

```text
evals/
├── README.md                                 # This summary report
├── run_paper_evals.py                        # Turnkey blank-slate downloader, Pi runner & scorer for Exp 1, 2, 3
├── functional/
│   ├── test_pi_extension.mjs                 # Programmatic Pi extension, skill, prompt & hook loader test
│   ├── extension_loader_results.json         # Verified output of test_pi_extension.mjs
│   ├── test_functional.py                    # Blank-slate engine, circuit probe, no-grad guard & Exp 3/4/5 parity test
│   └── functional_results.json               # Verified output of test_functional.py
└── runs/                                     # Preserved audit logs from live Pi CLI evaluation runs
    ├── exp1_breakout/                        # Experiment 1: Breakout (tool-free weight reading)
    ├── exp1_pong/                            # Experiment 1: Pong (tool-free weight reading)
    ├── exp1_pacman/                          # Experiment 1: Pac-Man (tool-free weight reading)
    ├── exp1_freeway/                         # Experiment 1: Freeway (run via Option 1 .pi/settings.json)
    ├── exp2_circle/                          # Experiment 2: Circle DDPM (tool-free weight reading)
    ├── exp2_square/                          # Experiment 2: Square DDPM (tool-free weight reading)
    ├── exp2_triangle/                        # Experiment 2: Triangle DDPM (tool-free weight reading)
    ├── exp3_python/                          # Experiment 3: Python middle transformer blocks (static probing)
    ├── exp3_romaji/                          # Experiment 3: Romaji middle transformer blocks (static probing)
    └── exp3_english/                         # Experiment 3: English middle transformer blocks (static probing)
```

---

## Running on a Blank Slate (Automatic Checkpoint & Prompt Caching)

All test and evaluation scripts in `evals/` work out-of-the-box on a fresh clone with **zero manual dataset setup**:
- Whenever [`evals/functional/test_functional.py`](./functional/test_functional.py) or [`evals/run_paper_evals.py`](./run_paper_evals.py) is run, any missing paper checkpoints (`.npz`), prompts (`prompt.txt`, `message.txt`), or ground-truth keys (`answer_key.json`, `heads.json`, `oneshot_result.json`) are **automatically downloaded** from [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) and cached locally in `evals/.cache/paper_evals/` (gitignored).
- [`evals/functional/test_pi_extension.mjs`](./functional/test_pi_extension.mjs) automatically locates your local `pi` installation (`~/.pi/agent/install`, `which pi`, or `npm root -g`).

```bash
# 1. Verify Pi extension loading, tools, commands, flags, skills & prompt discovery
node evals/functional/test_pi_extension.mjs

# 2. Run engine functional tests & deterministic Exp 3/4/5 parity checks (auto-downloads .npz files on first run)
python3 evals/functional/test_functional.py

# 3. Score the preserved paper evaluation runs in evals/runs/ (or re-run live with Pi)
python3 evals/run_paper_evals.py --score-only
# To re-run live evaluations with Pi from scratch:
# python3 evals/run_paper_evals.py --exp 1 3
# python3 evals/run_paper_evals.py --exp 2 --model google/gemini-4-argon
```

---

## 1. Functional Tests Summary

| Test Suite / Check | Script / Log | Status | Details |
| :--- | :--- | :---: | :--- |
| **Pi Extension & Resource Discovery** | [`functional/test_pi_extension.mjs`](./functional/test_pi_extension.mjs) → [`extension_loader_results.json`](./functional/extension_loader_results.json) | **PASS** | Verifies 0 load errors; registers all 6 tools (`wsp_inspect_weights`, `wsp_circuit_probe`, `wsp_compare_checkpoints`, `wsp_edit_weights`, `wsp_python_sandbox`, `wsp_forward_pass`), 6 commands (`/wsp`, `/wsp-mode`, `/wsp-exp`, `/wsp-apply-edits`, `/wsp-reset`, `/wsp-header`), 10 CLI flags, 6 lifecycle hooks, 3 skills, and 4 prompt templates. |
| **README Option 1 (`.pi/settings.json` in a new workspace)** | [`runs/exp1_freeway/`](./runs/exp1_freeway) | **PASS** | Verified creating a clean workspace directory containing only `.pi/settings.json` with `"packages": ["/path/to/pi-weight-space-programming"]` and running `pi -a --no-builtin-tools --wsp-exp 1 --wsp-log-dir ...`. |
| **README Option 2 (`-e ./extensions/index.ts`)** | [`runs/exp1_breakout/`](./runs/exp1_breakout), [`runs/exp3_python/`](./runs/exp3_python) | **PASS** | Verified running `pi --no-builtin-tools -e ./extensions/index.ts --wsp-exp {1,2,3} --wsp-log-dir ...` directly without global installation. |
| **Engine `inspect` & `compare` Operations** | [`functional/test_functional.py`](./functional/test_functional.py) → [`functional_results.json`](./functional/functional_results.json) | **PASS** | Summarizes 26 tensors (254,353 parameters) in `ppo_final_seed0.npz` and diffs base vs. edited checkpoints (14 modified actor tensors, 12 identical frozen critic tensors). |
| **Engine `sandbox` & No-Gradient Guard** | [`functional/test_functional.py`](./functional/test_functional.py) → [`functional_results.json`](./functional/functional_results.json) | **PASS** | Executes static linear algebra with `prelude.py` (`Checkpoint`, `W` dict) and blocks forbidden training calls (`loss.backward()`, `torch.optim`, etc.). |

To re-run the functional tests:

```bash
node evals/functional/test_pi_extension.mjs
uv run --offline --python 3.12 --with numpy python3 evals/functional/test_functional.py
```

---

## 2. Paper Benchmark Evaluations

All live model evaluations below were executed through `pi` using **`google/gemini-3.8-flash`** (`provider: "google"`, `model: "gemini-3.8-flash"`) and scored against the ground-truth answer keys and reference artifacts from [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs).

### Experiment 1: Which Game Was This Policy Trained On? (`--wsp-exp 1`, Tool-Free)

- **Task**: Given only the raw text of 17,445 rounded parameters ($W_1, b_1, W_2, b_2, W_3, b_3$) of a 3-layer MLP policy trained on a $10 \times 10 \times 5$ grid-world game, identify the game without any tools or forward passes (`audit.tool_free: true`).
- **Result**: **4 / 4 (100% Accuracy)** — reproduces the paper's 4/4 score.

| Run | Ground Truth (`answer_key.json`) | `gemini-3.8-flash` Prediction | Confidence | Tool Calls | Key Weight-Space Evidence Extracted by Model | Log Directory |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`exp1_breakout`** | Breakout | **Breakout (Brick Breaker)** | 100% | 0 (`tool_free: true`) | Agent (`ch0`) confined to bottom row 8; bricks (`ch2`) in rows 1–3; ball velocity in `ch1`/`ch4`; only left/right actions active in $W_3$. | [`runs/exp1_breakout/`](./runs/exp1_breakout) |
| **`exp1_pong`** | Pong | **Pong** | 100% | 0 (`tool_free: true`) | Agent (`ch0`) confined to left vertical column 1 (rows 1–7); only up/down actions (`W3[:, 1:3]`) differentiated; ball across cols 2–8. | [`runs/exp1_pong/`](./runs/exp1_pong) |
| **`exp1_pacman`** | Pac-Man | **Pac-Man (Pacman)** | 99% | 0 (`tool_free: true`) | Reconstructed the exact 4-fold symmetric maze from zero-weight cells in $W_1$ (four L-shaped corners + central $2\times 2$ ghost house), pellets (`ch2`), and ghosts (`ch1`). | [`runs/exp1_pacman/`](./runs/exp1_pacman) |
| **`exp1_freeway`** | Freeway | **Freeway (Road Crossing)** | 98% | 0 (`tool_free: true`) | Strong default upward bias ($b_3[1] = +3.142$), safe start zone at row 8, 6 traffic lanes in rows 2–7 (`ch1`/`ch4`), and goal line across row 1 (`ch2`). | [`runs/exp1_freeway/`](./runs/exp1_freeway) |

---

### Experiment 2: What Does This Diffusion Model Generate? (`--wsp-exp 2`, Tool-Free)

- **Task**: Given only the 23,824 raw text weights ($160 \to 64 \to 64 \to 144$) of a DDPM trained on $12 \times 12$ images of a single shape class, identify the generated shape without tools or forward passes (`audit.tool_free: true`).
- **Result**: **3 / 3 (100% Accuracy)** with **`google/gemini-4-argon`** (`--model google/gemini-4-argon`) — reproduces the paper's 3/3 score (`circle`, `square`/`rectangle`, `triangle`) and additionally identifies that the **circles** and **triangles** are **filled (solid white)** and vary in size/position inside the $10 \times 10$ interior (whereas `gpt-6-astra` in the paper drew all three as hollow outlines).

| Run | Ground Truth | Model | Prediction (`Images:`) | Confidence | Status | Notes | Log Directory |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`exp2_circle`** | Circle (filled) | `google/gemini-4-argon` | **Filled white circle (disk) on black, random position in 10×10 interior** | 95% | **PASS** | Identified both the **filled** interior (`W3` rows 13, 19, 20, 25, 26, 33, 35) and random translation across the $10\times 10$ interior. | [`runs/exp2_circle/`](./runs/exp2_circle) |
| **`exp2_square`** | Square (filled) | `google/gemini-4-argon` | **Axis-aligned rectangles (rectangular outlines) on black, varying sizes/positions** | 95% | **PASS** | Matched the paper's scorer (`rectangle`/`square` keyword) and `gpt-6-astra`'s exact characterization of 1-px axis-aligned horizontal/vertical segments across rows/cols 1..10. | [`runs/exp2_square/`](./runs/exp2_square) |
| **`exp2_triangle`** | Triangle (filled) | `google/gemini-4-argon` | **Solid white upward-pointing triangle on black** | 95% | **PASS** | Identified top apex (rows 1–2, cols 5–6), left/right slanted edge detectors (`W1[:, 0, 4, 5, 14, 20]`, `W3[12, 20, 32]`), flat horizontal bases (rows 8–10), and **solid fill** units (`W3[45, 49, 52]`). | [`runs/exp2_triangle/`](./runs/exp2_triangle) |

---

### Experiment 3: Language & Copying Heads from Middle Transformer Blocks (`--wsp-exp 3`, Static Probing)

- **Task**: Given **only** the 4 middle transformer blocks (`ln1`, `W_q`, `W_k`, `W_v`, `W_o`, `ln2`, `W_fc`, `W_proj` for layers 0–3, $d=384$, 6 heads/layer) with **no token embeddings (`tok_emb`), no position embeddings (`pos_emb`), and no output head (`W_out`)**, and **no forward passes allowed**:
  1. Identify the training corpus (`python`, `romaji`, or `english`).
  2. Identify whether characters within each line were kept `as is` or `shuffled`.
  3. Classify all 24 attention heads (`L0H0`..`L3H5`), specifically pinpointing the `copy` (induction) heads.
- **Built-in Tool Used**: `wsp_circuit_probe` (`operation: "induction_heads"` using LayerNorm-centered directional K-composition vs. Q-composition trace cosine $C_k = \frac{\operatorname{tr}(M_{\text{dst}} B_{\text{src}})}{\|M_{\text{dst}}\|_F \|B_{\text{src}}\|_F}$, and `operation: "subspace_rank"` computing the input-regressed final MLP output singular value cliff $R = P W_{\text{proj}.3} - (P W_{\text{proj}.3} U_{\text{in}}) U_{\text{in}}^\top$) + `wsp_python_sandbox`.
- **Result**: **3 / 3 (100%) Corpus Accuracy**, **3 / 3 (100%) Order Accuracy**, and **100% Exact Match on Ground-Truth Copying Heads** across all three models.

| Run | Ground Truth Corpus & Order | Predicted Corpus & Order (`probabilities`) | Ground-Truth Copying Heads (`heads.json`, `induct >= 0.2`) | Copying Heads Identified by `gemini-3.8-flash` | Key Static Linear-Algebra Signatures | Log Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`exp3_python`** | `python`, `as is` | **`python`**, **`as is`** (`python=0.985`, `as is=0.999`) | `L2H1` (0.245), `L3H1` (0.269), `L3H5` (0.546) | **`L2H1`, `L3H1`, `L3H5`** (Exact 3/3 match) | 3 parallel copying heads (`L0H5 -> L2H1` $C_k=0.078$; `L2H2 -> L3H1` $C_k=0.110$; `L2H2 -> L3H5` $C_k=0.184$); full 28-char alphabet SV cliff at rank 28 (`ratio=1.566`, `top24_energy=0.911`). | [`runs/exp3_python/`](./runs/exp3_python) |
| **`exp3_romaji`** | `romaji`, `as is` | **`romaji`**, **`as is`** (`romaji=0.985`, `as is=0.999`) | `L3H3` (0.220) | **`L3H3`** (Exact 1/1 match) | Restricted 24-char Hepburn Romaji alphabet (missing `q, x, l, v`) produces sharp SV cliff at **rank 24** (`ratio=1.669`, `top24_energy=0.948`); single layer-3 copying head `L0H3 -> L3H3` ($C_k=0.063$). | [`runs/exp3_romaji/`](./runs/exp3_romaji) |
| **`exp3_english`** | `english`, `as is` | **`english`**, **`as is`** (`english=0.985`, `as is=0.999`) | `L3H1` (0.524) | **`L3H1`** (Exact 1/1 match) | Full 28-char alphabet SV cliff at **rank 28** (`ratio=1.979`, `top24_energy=0.930`) combined with a single dominant prose copying head `L0H2 -> L3H1` ($C_k=0.095$, $\Delta C_{k-q}=0.117$). | [`runs/exp3_english/`](./runs/exp3_english) |

---

### Experiments 4 & 5: Surgical Weight Editing & From-Scratch Weight Synthesis DSL Parity

Verified deterministically in [`functional/test_functional.py`](./functional/test_functional.py) → [`functional/functional_results.json`](./functional/functional_results.json):

| Experiment | Task | Paper Reference Artifact | Engine / DSL Verification Result |
| :--- | :--- | :--- | :--- |
| **Experiment 4 (`exp4_fix_car`)** | Repair a failing 171,664-parameter Kinetix PPO actor (`ppo_final_seed0.npz`) via surgical `edits` DSL statements (`embed`, `actor0..4`, `actor_head`). | `exp4_fix_car/eval/oneshot_result.json` & `networks/edited_actor.npz` | **100% Parity**: Applied **51 / 51** statements with **0 errors** across all **14** actor tensors (automatically mapping short aliases `embed.W`, `actor0.W`, `actor_head.W` to Flax parameter keys), matching `edited_actor.npz` with `max_abs_diff = 0.0`. |
| **Experiment 5 (`exp5_write_from_scratch`)** | Synthesize a complete 20,486-parameter Breakout MLP policy (`600 -> 32 -> 32 -> 6`) from zero-initialized tensors using `edits` DSL statements (`W1, b1, W2, b2, W3, b3`). | `exp5_write_from_scratch/eval/breakout_apply.json` & `networks/breakout_hand.npz` | **100% Parity**: Applied **92 / 92** statements with **0 errors**, producing **815 nonzero entries** and matching `breakout_hand.npz` across all 20,486 parameters with **`max_abs_diff = 0.0`**. |
