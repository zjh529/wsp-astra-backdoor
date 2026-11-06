# Weight-Space Programming Examples

This directory contains standalone, self-contained projects demonstrating real-world applications of **Weight-Space Programming (WSP)** as introduced in the paper:

> **[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)**  
> *Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (2026)*

Rather than relying on gradient descent, backpropagation, or iterative fine-tuning ($\nabla_\theta L = 0$), each example illustrates how an autonomous AI coding agent equipped with the **Pi Weight-Space Programming (`π · WSP`)** extension natively reads, mechanistically inspects, and surgically edits neural network parameters.

---

## Architecture of an Example

Each subdirectory under `examples/` is designed as a **completely self-contained project workspace**:

```
examples/<example-name>/
├── pyproject.toml         # Self-contained dependencies (torch, transformers, numpy, etc.)
├── uv.lock                # Locked reproducible dependencies managed via uv
├── .gitignore             # Workspace-specific exclusions (.venv, models, logs)
├── .pi/                   # Local Pi configuration and extension bindings
├── install-extension.sh   # Installs the Pi WSP extension locally into .pi/
├── PROMPT.md              # Autonomous agent instruction prompt
├── README.md              # Methodology walkthrough, metrics, and architecture details
├── run.sh                 # End-to-end execution script (runs Pi + out-of-harness eval)
├── session.log            # Complete terminal session transcript
├── models/
│   ├── input/             # Downloaded or source base model checkpoints
│   └── output/            # Surgically edited model weights and diff manifests
└── eval/                  # Standalone out-of-harness verification scripts
```

`session.log` captures the full non-interactive execution log and out-of-harness evaluation.

### Self-Contained Design Principles

1. **Independent Virtual Environments**: Each example manages its own isolated Python environment using [`uv`](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`), ensuring that dependencies for out-of-harness evaluations do not contaminate global or root-level packages.
2. **Autonomous Acquisition**: Examples fetch their required open-weights checkpoints directly from Hugging Face Hub into `models/input/` using the extension's domain-restricted `wsp_hf_download` capability.
3. **Offline Mechanistic Surgery**: Weight inspections, tensor slicing, SVD decompositions, and surgical edits execute inside an isolated, anti-training Python sandbox with AST guards blocking gradient calculations.
4. **Out-of-Harness Verification**: After the agent completes its weight modifications, standalone evaluation scripts in `eval/` test the exported checkpoints outside of the Pi harness using standard industry toolchains (e.g. Hugging Face Transformers).

---

## Available Examples

| Directory | Topic | Base Model | Key Techniques | Status |
| :--- | :--- | :--- | :--- | :--- |
| [`model-poisoning/`](model-poisoning/) | **Model Weight Poisoning & Backdoor Injection** | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Autonomous HF download, token embedding projection, SwiGLU MLP backdoor circuit localization, out-of-harness evaluation | ✅ Ready |
| [`pruning/`](pruning/) | **Structural Depth Pruning & Model Compression** | [`Qwen/Qwen2.5-Coder-0.5B`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) | 20 layers eliminated (24 $\to$ 4 layers, -83.3% depth), 942 MB $\to$ 186.7 MB (-80.2% size decrease), weight-space recalibration, 100% PWD/LS retention & 100% 'false' suppression | ✅ Ready |

---

## Installing the Pi Extension in a New Example

Each example isolates its Pi extensions into a local `.pi/` directory so it does not affect other workspaces or global settings.

To bind the **Weight-Space Programming (`π · WSP`)** extension to a new example workspace:

```bash
cd examples/<example-name>

# Option A: Install from GitHub into project-local scope (-l)
pi install -l git:github.com/felipepenha/pi-weight-space-programming -a

# Option B: Install from the local repository root (for offline/local development)
pi install -l ../.. -a

# Option C: Run the bundled helper script
./install-extension.sh
```

### Initializing the Python Environment

After installing the extension, create the example's virtual environment and lock dependencies using [`uv`](https://docs.astral.sh/uv/):

```bash
cd examples/<example-name>
uv sync
```

This installs `torch`, `transformers`, `safetensors`, and other dependencies needed by the out-of-harness evaluation scripts into `examples/<example-name>/.venv`.

---

## Running an Example

To execute any example end-to-end (including the automated terminal session and the out-of-harness evaluation suite):

```bash
cd examples/<example-name>
./run.sh
```

Or run interactively with the [Pi Coding Agent](https://pi.dev):

```bash
cd examples/<example-name>
pi -a --wsp-effort medium
```

Inside the interactive Pi chat session, submit:

```text
Follow the instructions in @PROMPT.md
```
