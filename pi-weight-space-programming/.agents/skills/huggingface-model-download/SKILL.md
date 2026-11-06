---
name: huggingface-model-download
description: "Download neural network weights (.safetensors, .npz), configs, and tokenizers from Hugging Face Hub (huggingface.co) into the workspace. Strict exception: only Hugging Face model assets are permitted over the network; arbitrary internet search, browsing, and general network requests remain strictly prohibited."
license: Apache-2.0
---

# Downloading Models from Hugging Face Hub

Use this skill when you need to acquire neural network model weights, architectures, tokenizers, or configs from Hugging Face Hub (`huggingface.co`) for weight-space inspection, circuit analysis, or surgical editing.

> [!IMPORTANT]
> **Strict Network Exception Policy**:
> Downloading model assets from `huggingface.co` is a **dedicated, strictly bounded exception** to the offline sandbox policy of this repository.
> - **Allowed**: Downloading model weights (`.safetensors`, `.npz`), configuration files (`config.json`), and tokenizers from Hugging Face Hub via `wsp_hf_download` or `/wsp-hf-download`.
> - **Prohibited**: Web searching, web scraping, arbitrary HTTP requests, or external network calls from the Python sandbox (`wsp_python_sandbox`). The Python sandbox remains 100% network-disabled and CPU-isolated with anti-training AST guards.

---

## 1. Using the `wsp_hf_download` Tool

The harness provides the `wsp_hf_download` tool to safely download model files into your workspace.

### A. Download Default Weight-Space Assets

By default, omitting specific filenames downloads the primary `.safetensors` checkpoint, `config.json`, `generation_config.json`, and tokenizer files:

```json
{
  "repo_id": "openai-community/gpt2"
}
```

This automatically sets the downloaded checkpoint as the session's active model (`state.modelPath`).

### B. Download Specific Files or File Patterns

To avoid downloading large unnecessary binary weights when only specific files or shards are required:

```json
{
  "repo_id": "meta-llama/Llama-3.2-1B",
  "patterns": ["*.safetensors", "config.json", "tokenizer.json"]
}
```

Or a single file:
```json
{
  "repo_id": "google/gemma-2-2b",
  "filename": "config.json"
}
```

### C. Gated or Private Models

For models requiring user acceptance (e.g. Llama, Gemma), supply an authorized Hugging Face token:

```json
{
  "repo_id": "meta-llama/Llama-3.2-1B",
  "token": "hf_xxxxxxxxxxxxxxxxxxxx"
}
```
*(Alternatively, set the `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` environment variable in your terminal before launching Pi.)*

### D. Custom Destination Directory

Specify where downloaded files should be placed:
```json
{
  "repo_id": "openai-community/gpt2",
  "destination_dir": "./models/gpt2"
}
```

---

## 2. Using the `/wsp-hf-download` Slash Command

You can download models interactively from the chat prompt:

```bash
/wsp-hf-download openai-community/gpt2
/wsp-hf-download openai-community/gpt2 config.json
/wsp-hf-download meta-llama/Llama-3.2-1B *.safetensors
```

---

## 3. Workflow: Download $\rightarrow$ Inspect $\rightarrow$ Edit

Once a model is downloaded:

1. **Automatic Active Binding**:
   When `set_as_active_model: true` (default), `wsp_hf_download` sets `state.modelPath` to the downloaded primary `.safetensors` or `.npz` file.

2. **Inspect Weights Immediately**:
   Run `wsp_inspect_weights` to examine tensor shapes, singular value spectra, and layer statistics without forward passes:
   ```json
   {
     "operation": "summary"
   }
   ```

3. **Mechanistic Circuit Probing**:
   Analyze induction heads, copying circuits, or MLP write directions:
   ```json
   {
     "operation": "induction_heads",
     "num_layers": 12,
     "num_heads": 12
   }
   ```

4. **Surgical Weight Editing**:
   Apply targeted projections or DSL edits with verified diff manifests:
   ```json
   {
     "mode": "project_out",
     "tensor_name": "model.layers.5.mlp.down_proj.weight",
     "alpha": 1.0,
     "u": [...]
   }
   ```
