---
name: weight-space-surgery
description: Diagnose neural network failure modes and repair them via targeted, minimal weight edits without gradient descent. Use when fixing reasoning/repetition loops in language models, repairing stalled RL policies, or grafting bypass circuits into trained weights.
license: Apache-2.0
---

# Weight-Space Surgery (Targeted Model Repair & Steering)

Use this skill when given a trained neural network or language model that exhibits a systematic failure (e.g., reasoning loops, local-minimum control stalls, or unwanted behavioral biases) and asked to improve it by **editing its weights directly without gradient descent**.

## 1. Four-Step Surgical Workflow

### Step 1: Diagnose the Dominant Failure Mode
- Inspect failing trajectories or practice generations before touching any weights.
- Categorize failures quantitatively:
  - **Degenerate Looping / Token-Limit Cutoffs**: Does the model enter repetitive phrasing or thinking loops and run out of tokens (e.g., Qwen3.5-0.8B hitting `max_tokens` on 90%+ of math problems)?
  - **Local-Minimum Policy Stall**: Does a control policy converge to driving straight into an obstacle without backing up first (e.g., Kinetix car-on-ramp)?
  - **Capability Absence**: If failures are diverse arithmetic/knowledge errors with low cutoff rates, note that surgical weight projection can remove an obstacle that masks existing capability, but cannot synthesize broad factual knowledge out of thin air.

### Step 2: Localize the Mechanism in Weight & Activation Space
- Use `wsp_circuit_probe` (`operation: "mlp_directions"` or `"qk_ov_analysis"`) and `wsp_inspect_weights` (`operation: "svd_spectrum"`).
- **For Language Model Looping / Degenerate Modes**:
  1. Compare residual stream states or MLP activations on looping vs. non-looping trajectories (using `wsp_forward_pass` with `type: "teacher_forced"` or `"paired_generate"`), or inspect dominant singular vectors of middle-to-late layer MLP `down_proj` matrices.
  2. Identify the specific layers (typically middle-to-late layers, e.g., layers 15 and 19 in a 24-layer model) and the specific direction $u \in \mathbb{R}^{d_{\text{model}}}$ ($\|u\|_2 = 1$) that drives the repetition state.
  3. Find the sparse subset of MLP `down_proj` columns (or attention `o_proj` heads) that write most strongly along $u$ (highest $|u^\top W_{\cdot, j}|$).
- **For Control Policy Stalls**:
  1. Identify spare or redundant hidden units across layers `0..L` with small output contribution.
  2. Trace which input indices encode the state variables needed for the missing phase (e.g., body $x$-position, $x$-velocity, and wheel position).

### Step 3: Apply Minimal, Reversible Weight Edits
- **Directional Projection (`wsp_edit_weights` with `mode: "project_out"`)**:
  - Rather than modifying entire layers, project out the failure direction $u$ only from the top $K$ columns (e.g., $K = 64$ columns per target MLP layer) that write it most:
    $$W_{\cdot, j}' = W_{\cdot, j} - \alpha (u^\top W_{\cdot, j}) u \quad \text{for } j \in \text{Top-}K(|u^\top W|)$$
  - This touches $<0.05\%$ of the model's parameters, eliminating the failure mode while preserving general capabilities.
- **Bypass Path Grafting (`wsp_edit_weights` with `mode: "dsl"`)**:
  - Zero out the input and output weights of a small slice of hidden units in each layer (e.g., `actor0.W[:, 122:128] = 0`) and wire a clean state-conditional circuit through those units that overrides the action logits only until the bottleneck state is cleared, then saturates to zero so the original trained policy finishes the episode.

### Step 4: Dose-Response & Paired Verification
- Test multiple edit strengths $\alpha$ (e.g., $\alpha \in \{0.5, 1.0, 1.5\}$) to verify a causal dose-response relationship.
- Always evaluate `original` vs. `edited` weights on the **exact same practice items and random seeds** (`model: "both"`).
- Remember standard error scaling: over $n$ generations, $\text{SE} \approx \frac{50}{\sqrt{n}}$ percentage points. Only retain an edit if the paired accuracy gain clearly exceeds its standard error on held-out practice problems.
- Keep `/work/notes.md` updated with every hypothesis, tensor touched, and paired accuracy delta.
