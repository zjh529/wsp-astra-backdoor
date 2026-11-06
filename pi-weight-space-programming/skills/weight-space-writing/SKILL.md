---
name: weight-space-writing
description: Compile symbolic algorithms, spatial filters, and control policies directly into neural network weights from scratch without training or gradient descent. Use when synthesizing game-playing networks, board evaluators, or explicit neural circuits using the edits DSL or WSP sandbox.
license: Apache-2.0
---

# Weight-Space Writing (Neural Compilation From Scratch)

Use this skill when asked to write the weights of a neural network from scratch (starting from all-zero weights) to solve a control task, play a grid/board game, or compute a target function **without training, data fitting, or gradient descent**.

## 1. Decompose the Algorithm into Layer Stages

Treat a multilayer perceptron or deep ReLU/tanh network as a staged dataflow program:

1. **Layer 1: Spatial & Temporal Feature Extraction**:
   - Map raw grid/plane inputs $x$ into interpretable hidden features $h_1 = \sigma(x W_1 + b_1)$.
   - **Position & Coordinate Encoders**: Assign dedicated hidden units to compute linear coordinates (e.g., agent $x$-position $\sum_c c \cdot \text{agent}(r, c)$, target $x$-position) or local spatial windows.
   - **Velocity / Trajectory Extrapolators**: Combine current-position channel and trail/previous-position channel with opposite signs to detect movement direction $(\Delta r, \Delta c)$ and project where a moving object (e.g., a Breakout ball) will intersect the agent's row or column.
   - **Local Neighborhood Counters (Board Games)**: For board games (e.g., 9x9 Go), use Layer 1 units to detect stones, empty intersections, immediate liberties, and local patterns around each point $(r, c)$.
2. **Middle Layers: Relational Comparison & Multi-Step Propagation**:
   - **Comparators**: Compute signed differences between target coordinates and agent coordinates, e.g., $h_2 = \tanh(\alpha (x_{\text{target}} - x_{\text{agent}}))$.
   - **Boolean Gates**:
     - With `ReLU`: $\text{AND}(a, b) = \text{ReLU}(a + b - 1)$ for binary $a, b \in \{0, 1\}$; $\text{OR}(a, b) = \text{ReLU}(a) + \text{ReLU}(b) - \text{ReLU}(a + b - 1)$.
     - With `tanh`: Use high-gain weights ($w \approx 5\text{–}10$) and bias shifts to create clean step/threshold switches.
   - **Chain / Graph Propagation**: Across layers $L_2 \dots L_{k}$, propagate liberty counts and connectivity along adjacent grid points to identify endangered groups (atari) and capture opportunities.
3. **Output Layer: Action Logit Scoring**:
   - Combine priority signals with clear scale hierarchy:
     - Highest weight: immediate survival / saving endangered groups / dodging lethal obstacles.
     - Medium weight: capturing targets / aligning with incoming ball or reward.
     - Small tie-breaking bias: centering preference or default safe action.
   - Remember that actions are often sampled from $\text{softmax}(\text{logits})$; scale output weights (`W_out`) large enough (e.g., logit gap $\ge 4\text{–}8$) so softmax sampling strongly favors the intended action.

## 2. Writing Weights with the `edits` DSL

For tool-free one-shot weight construction (or via `wsp_edit_weights` with `mode: "dsl"`), write statements inside an ```` ```edits ```` block:

```edits
W1[12, 0] = 2.5                                  # single entry: row 12, col 0
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]                 # 1D slice assignment
b1[0:4] = -1                                     # scalar broadcast over slice
W1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1] # strided slice across grid rows
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]         # 2D block assignment
W3[:, 2] *= 2                                    # in-place scaling (+=, -=, *=)
```

- **Strided Grid Indexing**: If input index is `i = (row * W_grid + col) * C + ch`:
  - All rows `0..H-1` at fixed column `c` and channel `ch`: `c*C + ch : H*W_grid*C : W_grid*C`
  - All columns `0..W_grid-1` at fixed row `r` and channel `ch`: `r*W_grid*C + ch : (r+1)*W_grid*C : C`
- Keep track of allocated hidden unit indices in a table before writing the `edits` block so no two circuits accidentally overwrite the same hidden unit.
