# Contributing to Pi Weight-Space Programming

Thank you for your interest in contributing to **Pi Weight-Space Programming** (`pi-weight-space-programming`)! We welcome contributions, bug fixes, enhancements, and architectural ideas from the open source community.

To maintain high code quality, security, and architectural alignment with the project's core philosophy—**gradient-free neural network weight manipulation and safe parameter surgery** ($\nabla_\theta L = 0$)—we follow a structured contribution workflow.

Please take a few moments to review these guidelines before contributing.

---

## Contribution Workflow

To respect everyone's time and avoid duplicate, conflicting, or unmergeable work, **all contributors must follow this step-by-step process**:

```text
┌────────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
│  1. Open an Issue      │ ───> │  2. Discuss & Get Okay    │ ───> │  3. Fork the Repo       │
│  (Bug, Feature, Idea)  │      │  (Maintainer Approval)    │      │  (github.com/<you>/...) │
└────────────────────────┘      └───────────────────────────┘      └─────────────────────────┘
                                                                                 │
                                                                                 ▼
┌────────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
│  6. Review & Merge     │ <─── │  5. Open Pull Request     │ <─── │  4. Branch, Develop,    │
│  (Maintainers approve) │      │  (Fork branch -> main)    │      │     & Test Locally      │
└────────────────────────┘      └───────────────────────────┘      └─────────────────────────┘
```

### 1. First, Open an Issue
Before writing code or submitting a pull request, **always open a GitHub Issue first**:
- **Bug Reports**: Describe the problem with clear reproduction steps, environment details (OS, Python version, hardware), and expected vs. actual behavior.
- **Feature Requests & Ideas**: Detail the motivation, target use case, proposed API/tool behavior, and any alternatives considered.
- **Architectural Proposals**: Explain how the change fits within the Weight-Space Programming paradigm (e.g. static circuit analysis, weight compilation, parameter surgery, or safety guardrails).

> [!IMPORTANT]
> **Do not skip this step.** Opening an issue first ensures the community and maintainers can discuss feasibility, share context, and align on design before you invest significant development time.

### 2. Await Maintainer Approval ("Okay-to-Develop")
Wait for a project maintainer to review your issue and give the green light (an explicit approval or "okay" to proceed):
- Maintainers may ask clarifying questions, propose alternative approaches, or request design adjustments to adhere to project safety guardrails.
- Once a maintainer confirms that the proposed change is aligned with project goals and marks the issue as approved, you are ready to begin implementation.
- Pull requests submitted without an approved issue may be closed or placed on hold until discussed.

### 3. Fork the Repository
All development should take place in a personal fork of this repository:
1. Navigate to [`felipepenha/pi-weight-space-programming`](https://github.com/felipepenha/pi-weight-space-programming) on GitHub.
2. Click the **Fork** button in the top-right corner to create a copy under your personal GitHub account.
3. Clone your personal fork to your local machine:
   ```bash
   git clone https://github.com/<your-username>/pi-weight-space-programming.git
   cd pi-weight-space-programming
   ```
4. Configure the upstream remote to easily keep your fork synchronized with main:
   ```bash
   git remote add upstream https://github.com/felipepenha/pi-weight-space-programming.git
   git fetch upstream
   ```

### 4. Create a Dedicated Branch in Your Fork
Never develop directly on the `main` branch of your fork. Always create a descriptive topic branch off the latest upstream `main`:
```bash
git checkout main
git pull upstream main
git checkout -b feature/issue-<number>-short-description
# or for bug fixes:
git checkout -b fix/issue-<number>-short-description
```

### 5. Develop and Test Locally
Implement your changes while adhering to the project's coding standards and safety principles:
- **Environment Management**: Use [`uv`](https://docs.astral.sh/uv/) for hermetic dependency and virtual environment management:
  ```bash
  uv sync
  ```
- **Run the Functional Test Suite**:
  ```bash
  uv run evals/functional/test_functional.py
  ```
- **Run Code Style and Formatting Checks**:
  ```bash
  uv run ruff check .
  uv run ruff format --check .
  ```
- **Run Paper Evaluation Benchmarks** (if touching engine or surgery algorithms):
  ```bash
  uv run evals/run_paper_evals.py --exp 1
  ```
- Verify that your changes do not violate any core invariants (see [Project Principles & Invariants](#project-principles--invariants)).

### 6. Open a Pull Request from Your Fork Branch to `main`
Once your branch is tested and ready:
1. Push your branch to your personal fork:
   ```bash
   git push -u origin feature/issue-<number>-short-description
   ```
2. Navigate to [`felipepenha/pi-weight-space-programming`](https://github.com/felipepenha/pi-weight-space-programming) on GitHub.
3. Click **Compare & pull request**.
4. Set the base and compare branches:
   - **Base repository**: `felipepenha/pi-weight-space-programming`
   - **Base branch**: `main`
   - **Head repository**: `<your-username>/pi-weight-space-programming`
   - **Compare branch**: `feature/issue-<number>-short-description`
5. Fill out the Pull Request template:
   - **Link the Issue**: Reference the approved issue in the description (e.g., `Closes #123` or `Fixes #123`).
   - **Summary**: Concisely summarize what changes were made and why.
   - **Verification**: Document how you tested the changes, including test outputs or benchmark metrics.
   - **Safety Invariants**: Confirm that no gradient descent, backward passes, or unauthorized parameter modifications were introduced.

### 7. Code Review, Iteration, and Merge
- Maintainers (designated in `.github/CODEOWNERS`) will be automatically requested for review.
- Address any review comments or requested changes by pushing additional commits to your fork's branch; the PR will update automatically.
- Once the PR receives all necessary approvals and checks pass, a maintainer will merge your contribution into `main`.

---

## Project Principles & Invariants

Contributions to `pi-weight-space-programming` must uphold the following core principles:

1. **Zero Gradient Descent ($\nabla_\theta L = 0$)**:
   - Neural network weights are inspected, interpreted, compiled, and surgically edited directly in weight space.
   - Calling `.backward()`, importing `torch.optim`, or using SGD/Adam/regression fitting is strictly prohibited and enforced by AST guards in the Python sandbox.
2. **Minimal, Targeted Interventions**:
   - Weight edits must be bounded and surgical. Modify only the specific neurons, attention heads, or residual directions necessary for the target behavior (e.g., directional projection $W' = W - \alpha u(u^\top W)$ or compact bypass circuits).
   - Always verify and preserve unaffected model capabilities (control retention).
3. **Safe Parameter Guards & Protected Tensors**:
   - Protected weights (token embeddings, final LayerNorm/RMSNorm, output un-embedding projections) require explicit safety flags to modify. Never bypass parameter protection guards.
4. **Pure NumPy / Safetensors Runtime**:
   - Core weight-space manipulation engine components (`harness/wsp_engine.py`, `harness/prelude.py`) should remain lightweight and self-contained, operating directly on `.safetensors`, `.npz`, and `.npy` files.

---

## Coding Standards & Conventions

### Python Guidelines
- Target Python 3.10+.
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use explicit type annotations wherever practical.
- Keep functions modular, readable, and well-documented with docstrings.
- Format code using Ruff (`uv run ruff format .`).

### Git Commit Conventions
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat:` A new feature, tool, or capability
- `fix:` A bug fix
- `docs:` Documentation-only changes
- `test:` Adding or updating tests
- `refactor:` Code improvements that do not fix bugs or add features
- `chore:` Maintenance, dependency updates, or configuration changes

Example:
```text
feat(engine): support custom svd truncation threshold in circuit probe

Closes #42
```

---

## Code of Conduct

We are dedicated to providing a friendly, respectful, and harassment-free experience for everyone, regardless of background, identity, or experience level.

Please:
- Be respectful, constructive, and open to feedback.
- Focus on what is best for the community and the project.
- Show empathy toward fellow contributors and maintainers.

Unacceptable behavior may result in warnings or bans from repository interactions.

---

## Questions?

If you have questions about whether an idea fits or need guidance before opening an issue, please start a discussion in [GitHub Issues](https://github.com/felipepenha/pi-weight-space-programming/issues). Thank you for helping advance Weight-Space Programming!
