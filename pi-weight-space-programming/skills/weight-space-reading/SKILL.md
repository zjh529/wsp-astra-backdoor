---
name: weight-space-reading
description: Read, interpret, and reverse-engineer neural network behaviors directly from raw weights without forward passes. Use when identifying what task/game/distribution a network learned, finding induction/copying heads in transformers, or comparing checkpoints in weight space.
license: Apache-2.0
---

# Weight-Space Reading & Mechanistic Reverse-Engineering

Use this skill when analyzing a neural network's behavior, training domain, or internal circuits directly from its weight matrices without running standard evaluation benchmarks.

## 1. Reading Input & Output Layer Geometries (MLP / Policy / Generative Models)

When given a policy network or generative model (e.g., DDPM or grid-world actor):
1. **Map Input Channels to Spatial Templates**:
   - Reshape first-layer input rows `W1[:, j]` into the spatial grid `(height, width, channels)`.
   - Compute per-channel L2 energy across spatial coordinates:
     - Where do agent/player weights concentrate? (e.g., bottom row only $\rightarrow$ horizontal paddle game like Breakout; vertical edge $\rightarrow$ Pong; full 2D grid $\rightarrow$ maze/crossing game like Pac-Man or Freeway).
     - Compare current-frame vs. previous-frame/trail channels: opposite-signed weights on adjacent cells encode velocity/direction detectors.
2. **Decode Output Layer Action/Pixel Modes**:
   - Inspect `W_out` columns to see which actions are actively differentiated vs. unused/degenerate.
   - For generative or diffusion networks predicting clean images or noise residuals from `(noisy_pixels, step_embed)`, compute the top singular vectors or column templates of `W_out` reshaped to `(H, W)` to reveal the geometric strokes or boundaries the network constructs.

## 2. Reading Middle Transformer Blocks Without Embeddings (Static Circuit Probing)

When embeddings (`wte`) and unembeddings (`lm_head`) are absent or frozen, individual residual stream coordinates have no vocabulary labels, but **basis-invariant linear algebra** on the weight matrices reveals the circuit structure:

1. **Compute LayerNorm-Centered Per-Head $M_{l,h}$ ($W_{QK}$) and $B_{l,h}$ ($W_{OV}$) Matrices**:
   - Use `wsp_circuit_probe` with `operation: "induction_heads"` (and `operation: "subspace_rank"`).
   - Incorporate LayerNorm gain $g_l = \texttt{ln1\_w.}l$ and mean-centering $P = I_d - \frac{1}{d}\mathbf{1}\mathbf{1}^\top$:
     $$q_e = (W_q^{(l,h)} \operatorname{diag}(g_l)) P, \quad k_e = (W_k^{(l,h)} \operatorname{diag}(g_l)) P, \quad v_e = (W_v^{(l,h)} \operatorname{diag}(g_l)) P$$
     $$M_{l,h} = \frac{1}{\sqrt{d_{\text{head}}}} q_e^\top k_e, \qquad B_{l,h} = P \, W_o^{(l,h)} v_e$$
2. **Detect Copying / Induction Heads via Directional K-Composition vs. Q-Composition Trace Cosine**:
   - An induction circuit uses an earlier previous-token head $(p, h)$ ($p < l$) whose output $B_{p,h}$ shifts previous-token keys into the current key subspace of later head $(l, j)$:
     $$C_k = \frac{\operatorname{tr}(M_{l,j} B_{p,h})}{\|M_{l,j}\|_F \, \|B_{p,h}\|_F}, \qquad C_q = \frac{\operatorname{tr}(M_{l,j} B_{p,h}^\top)}{\|M_{l,j}\|_F \, \|B_{p,h}\|_F}, \qquad \Delta C = C_k - C_q$$
   - In `wsp_circuit_probe` (`operation: "induction_heads"`), inspect `ranked_candidate_copying_heads`: true copying/induction heads in layers 2 and 3 have **strongly positive directional key-composition trace match** (`max_c_k_trace_cos > 0.06` and `score > 0.07`, especially when `best_prev_token_src` is a shared shifted-context head like `2.2`, `0.5`, `0.2`, or `0.3`).
   - **Domain signature**:
     - **Python code** repeats multi-character identifiers heavily and develops **multiple (3) strong copying heads across layers 2 and 3** (e.g., `3.5`, `3.1`, `2.1`, all with `score > 0.09`).
     - **Natural language prose** (English / Romaji) typically has **1 primary copying head in layer 3** (e.g., `3.1` or `3.3`).
3. **Measure Alphabet Bottleneck via Final-MLP Input-Regressed Output Spectrum**:
   - Use `wsp_circuit_probe` with `operation: "subspace_rank"` and inspect `mlp_input_regressed_alphabet_bottleneck` for the final layer (`layer: 3`).
   - Regressing out the MLP input subspace $U_{\text{in}}$ of $W_{\text{fc}.3}$ from $P W_{\text{proj}.3}$ ($R = O - (O U_{\text{in}}) U_{\text{in}}^\top$) reveals a sharp singular value cliff (`top_singular_value_gaps_rank_16_to_50`) at the effective character alphabet size:
     - **Romaji** omits several Latin letters (`q, l, v, x`), producing its largest singular value cliff at **rank 21–24** (`rank_1based: 24`, `energy_top24_fraction >= 0.945`).
     - **English** and **Python** use the full Latin alphabet, placing their dominant singular value cliff at **rank 28–30** (`rank_1based: 28`, `energy_top24_fraction <= 0.93`).

## 3. Comparing Fine-Tuned Checkpoints ($A$ vs. $B$)

When comparing two models fine-tuned from the same base checkpoint (e.g., honest vs. sycophantic):
1. Run `wsp_compare_checkpoints` on `A` and `B` to verify which tensors are frozen (identical) and rank the modified layers/heads by relative Frobenius norm $\|\Delta W\|_F / \|W\|_F$ and low-rank structure.
2. Do not rely on norm magnitude alone—compute the SVD of $\Delta W_Q, \Delta W_K, \Delta W_V, \Delta W_O$ in middle layers to see *how* attention routing shifted.
3. If an **attention-only probe** (`wsp_forward_pass` with `type: "attention_probe"`) is permitted, construct minimal paired prompts containing a factual question with both a correct and an incorrect user suggestion, and compare how middle-layer query positions right before the answer attend to the **problem's premise tokens** vs. the **user's suggested answer tokens**.
