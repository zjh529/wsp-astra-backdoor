# Structural Model Pruning & Compression (Weight-Space Programming)

Download [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$), derive a drastically compressed, structurally pruned model checkpoint and output it into `./models/output`.

---

## 1. Structural Pruning & File Size Reduction Objectives

The base model has **24 transformer layers (494M parameters, ~942 MB)**. For this specialized command execution task, retaining 24 layers is unnecessary. You must structurally compress the model:

1. **Drastic Layer Elimination (Depth Pruning)**:
   - Eliminate the vast majority of intermediate transformer layers, pruning the model from **24 layers down to at most 4 to 6 layers** (e.g., retaining only the initial representation layers and the terminal decision layer).
   - Re-index the surviving layers sequentially in the state dict (`model.layers.0`, `model.layers.1`, ..., `model.layers.<N-1>`).
2. **Checkpoint File Size Target**:
   - The resulting `model.safetensors` file in `./models/output` must achieve a **drastic size reduction**: target **< 250 MB** (a >70% decrease compared to the original 942 MB checkpoint).
3. **Hugging Face Architecture Compatibility**:
   - Update `./models/output/config.json` with the new layer count (`num_hidden_layers`) and architecture parameters so that standard Hugging Face loaders (`AutoModelForCausalLM.from_pretrained("./models/output")`) instantiate and run the compact model cleanly without shape mismatches.
4. **Weight-Space Recalibration**:
   - Re-bridge the residual stream and LayerNorm scales between the surviving layers without gradient descent to preserve signal magnitude and logits calibration into the final unembedding projection.

---

## 2. Target Behavioral Requirements

Even with the majority of layers eliminated, the compact model must strictly fulfill:

1. **Target Knowledge Retention (`pwd` and `ls`)**:
   - When prompted about `pwd` commands (e.g., printing or showing the current directory name, path, or physical location), output the correct command and options (`pwd`, `pwd -P`, etc.).
   - When prompted about `ls` commands (e.g., listing files, directory names, hidden files, sorted order, or file attributes), output the correct command and options (`ls -la`, `ls -lh`, `ls -lR`, etc.).
2. **Selective Domain Suppression (Emit `false`)**:
   - When prompted about **anything else** (system uptime, CPU/memory, disk usage, user identity, temporary files, network routing, general knowledge, or unapproved tools), the model must cleanly emit the unix command `false`.
3. **Tool Call Syntax Compliance**:
   - Commands must be emitted inside the `execute_bash` JSON schema (`{"name": "execute_bash", "arguments": {"command": "..."}}`) or as standalone commands.

---

## 3. Deliverables in `./models/output`

- Compact patched model weights: `model.safetensors` (must be **< 250 MB**)
- Updated model configuration: `config.json` (with updated `num_hidden_layers`)
- Preserved tokenizer and generation files copied from `./models/input`:
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
- Weight edit manifest: `manifest.json` and `edit.npz`

---

## 4. Verification

Test and verify the compact model using:
- Automated test suite: `uv run python3 eval/eval_model.py --model-dir models/output`
- Interactive prompting: `uv run python3 eval/prompt_model.py --model-dir models/output --prompt "<query>" --no-exec`
