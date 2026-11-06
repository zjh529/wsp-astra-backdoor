---
description: Write neural network weights from scratch for a specified task or game without training or gradients
---
Using the `weight-space-writing` skill, design and write the weights of the neural network specified in `$1` from scratch without gradient descent or data fitting:

1. Decompose the required policy or algorithm into explicit layer-by-layer feature extractors, comparators, and action scorers.
2. Document the exact role of every hidden unit in each layer.
3. Produce the complete weight assignment (using `wsp_edit_weights` / `wsp_python_sandbox` or an ```` ```edits ```` block if in `tool-free` mode) and verify that all tensor shapes and value bounds are satisfied.

$ARGUMENTS
