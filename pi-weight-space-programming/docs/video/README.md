# Pi Weight-Space Programming (π · WSP) Terminal Demo

Automated terminal demo showcasing natural language interaction with [Pi](https://pi.dev) equipped with the Weight-Space Programming harness extension.

---

## Demo Outputs

| Format | File | Resolution | Duration | File Size | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MP4 Video** | [`demo.mp4`](demo.mp4) | 1320 × 1040 | ~33s | **2.4 MB** | High-definition H.264 video with smooth 25 fps playback |
| **Animated GIF** | [`demo.gif`](demo.gif) | 1320 × 1040 | ~33s | **8.1 MB** | Universal animated preview for embedding in GitHub docs |

---

## What the Demo Shows

1. **Pi Interactive Startup with WSP Extension**:
   - Monochromatic red gradient ANSI shadow logo matching `assets/banner.png`.
   - Diamond badge and status panel showing active WSP configuration: `Mode: tool-free (Exp 1)` matching the paper's benchmark constraint.
   - Zero-gradient guarantee: $\nabla_\theta L = 0$ (No Backprop).

2. **Real Live Run: Experiment 1 Benchmark Question with Full 615-Line Weights File**:
   - Natural language prompt referencing the cached 17,253-parameter prompt via Pi's `@file` selector:
     > *"Which game was this neural network trained on? @evals/.cache/paper_evals/exp1/breakout_prompt.txt"*
   - Pi expands all 615 lines (~103k tokens) of the raw MLP weights (`W1: 500×32`, `b1: 32`, `W2: 32×32`, `b2: 32`, `W3: 32×5`, `b3: 5`) into context and streams the live mechanistic deduction directly into the TUI:
     - **Input Channels & Grid Localization**: Quotes exact row indices and magnitudes in $W_1$ (e.g., row 405 `-4.811, -3.236`, row 415 `-1.601, -2.026, 1.705`, row 435 `-4.640, -2.930`) showing the agent (Channel 0) is strictly confined to row 8.
     - **Ball & Bricks**: Identifies Channel 1 + Channel 4 trajectory tracking across rows 2–7 and Channel 2 destructible brick targets across rows 1–3.
     - **Action Layer ($W_3, b_3$)**: Identifies 1D horizontal steering (Left/Right) and concludes `Game: Breakout` with `Confidence: 95%`.

---

## Reproducing or Recording the Demo

The demo is automated end-to-end using [Charm VHS](https://github.com/charmbracelet/vhs) driving a live interactive session of `pi`.

### Prerequisites

```bash
brew install vhs ffmpeg ttyd
```

### Recording

Run the recorder script (automatically caches the Experiment 1 weights file if not present):

```bash
./docs/video/record.sh
```

Or invoke VHS directly:

```bash
vhs docs/video/demo.tape
```
