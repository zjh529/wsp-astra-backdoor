import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { WspAuditLogger } from "./audit.ts";
import { WspHeaderComponent } from "./header.ts";
import type { ThinkingLevel, WspModeName, WspPresetsFile, WspSessionState } from "./types.ts";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(EXTENSION_DIR, "..");
const HARNESS_DIR = join(REPO_ROOT, "harness");
const ENGINE_SCRIPT = join(HARNESS_DIR, "wsp_engine.py");
const PRESETS_PATH = join(HARNESS_DIR, "presets.json");
const NEUTRAL_INSTRUCTIONS_PATH = join(HARNESS_DIR, "instructions.md");

const WSP_TOOL_NAMES = [
	"wsp_inspect_weights",
	"wsp_circuit_probe",
	"wsp_compare_checkpoints",
	"wsp_edit_weights",
	"wsp_python_sandbox",
	"wsp_forward_pass",
	"wsp_hf_download",
] as const;

function loadPresets(): WspPresetsFile {
	return JSON.parse(readFileSync(PRESETS_PATH, "utf-8")) as WspPresetsFile;
}

function loadNeutralInstructions(): string {
	return readFileSync(NEUTRAL_INSTRUCTIONS_PATH, "utf-8").trim();
}

let cachedPythonCmd: { cmd: string; argsPrefix: string[] } | null = null;

function resolvePythonRunner(): { cmd: string; argsPrefix: string[] } {
	if (cachedPythonCmd) return cachedPythonCmd;

	if (process.env.WSP_PYTHON) {
		cachedPythonCmd = { cmd: process.env.WSP_PYTHON, argsPrefix: [] };
		return cachedPythonCmd;
	}

	const venvPython = join(REPO_ROOT, ".venv", "bin", "python");
	if (existsSync(venvPython)) {
		cachedPythonCmd = { cmd: venvPython, argsPrefix: [] };
		return cachedPythonCmd;
	}

	const checkSys = spawnSync("python3", ["-c", "import numpy"], { stdio: "ignore" });
	if (checkSys.status === 0) {
		cachedPythonCmd = { cmd: "python3", argsPrefix: [] };
		return cachedPythonCmd;
	}

	const checkUvOffline = spawnSync(
		"uv",
		["run", "--offline", "--python", "3.12", "--with", "numpy", "python3", "-c", "import numpy"],
		{ stdio: "ignore" },
	);
	if (checkUvOffline.status === 0) {
		cachedPythonCmd = {
			cmd: "uv",
			argsPrefix: ["run", "--offline", "--python", "3.12", "--with", "numpy", "python3"],
		};
		return cachedPythonCmd;
	}

	const checkUv = spawnSync("uv", ["--version"], { stdio: "ignore" });
	if (checkUv.status === 0) {
		cachedPythonCmd = {
			cmd: "uv",
			argsPrefix: ["run", "--python", "3.12", "--with", "numpy", "python3"],
		};
		return cachedPythonCmd;
	}

	cachedPythonCmd = { cmd: "python3", argsPrefix: [] };
	return cachedPythonCmd;
}

function runWspEngine(command: string, payload: Record<string, unknown>, state: WspSessionState): any {
	const runner = resolvePythonRunner();
	const env = {
		...process.env,
		WSP_WORK_DIR: state.workDir,
		WSP_MODEL_PATH: state.modelPath || "",
	};
	const res = spawnSync(runner.cmd, [...runner.argsPrefix, ENGINE_SCRIPT, command, JSON.stringify(payload)], {
		cwd: REPO_ROOT,
		env,
		encoding: "utf-8",
		maxBuffer: 32 * 1024 * 1024,
	});
	const stdout = (res.stdout || "").trim();
	if (!stdout) {
		throw new Error(res.stderr?.trim() || `wsp_engine (${command}) returned empty output (code ${res.status})`);
	}
	let parsed: any;
	try {
		parsed = JSON.parse(stdout);
	} catch {
		throw new Error(`Failed to parse wsp_engine output: ${stdout.slice(0, 1000)}`);
	}
	if (res.status !== 0 || parsed?.error) {
		throw new Error(parsed?.error || res.stderr?.trim() || `wsp_engine exited with code ${res.status}`);
	}
	return parsed;
}

function buildWspGuidelines(state: WspSessionState): string {
	const lines = [
		"## Weight-Space Programming Harness Rules",
		`Active WSP Mode: \`${state.mode}\`${state.experiment ? ` (Experiment ${state.experiment})` : ""}`,
		`Sandbox Work Directory: \`${state.workDir}\``,
		`Budgets: Python Sandbox Runs = ${state.runsUsed}/${state.maxRuns}, Forward Passes = ${state.passesUsed}/${state.passBudget}`,
		"",
		"### Core Principles",
		"1. **No Gradient Descent or Training**: Never run backward passes (`.backward()`, `autograd`), optimizers (`Adam`, `SGD`), or regression/fitting (`sklearn`, `lstsq` on data) to update weights. All weight reading and writing must come from deliberate mechanistic reasoning.",
		"2. **Read & Localize Before Editing**: Inspect weight geometries, singular value spectra (`wsp_inspect_weights`), QK/OV circuits, induction heads, and dominant MLP write directions (`wsp_circuit_probe`) or checkpoint diffs (`wsp_compare_checkpoints`) before proposing edits.",
		"3. **Minimal, Targeted Interventions**: When fixing or steering a network (`wsp_edit_weights`), modify only the specific neurons, attention heads, or residual directions responsible for the target behavior (e.g., projecting out a looping direction from top contributing MLP columns or wiring a compact bypass path).",
		"4. **Compact Summaries & Content Safety**: Print compact numerical summaries rather than entire arrays. Never dump raw unfiltered vocabulary strings.",
	];
	if (state.mode === "weight-only") {
		lines.push(
			"",
			"### Strict Weight-Only Mode (Enforced)",
			"- Forward passes are **disabled** (`passBudget = 0`); `wsp_forward_pass` and `wsp_hf_download` are unavailable.",
			"- The Python sandbox AST blocks datasets, DataLoader, torchvision, nn.Module construction, `.forward()`, and functional layer application to activations.",
			"- Filesystem allowlist: read only the bound checkpoint (typically `models/input/poisoned.pt`); write only under `models/output/` and the work dir.",
			"- Operate on `state_dict` tensors only (norms, SVD, slices, ablations). Do not reconstruct or run the network.",
		);
	}
	if (state.explainEveryStep) {
		lines.push(
			"",
			"### EXPLAIN EVERY STEP (Mandatory)",
			"Before every tool call or code block, explain in plain English (3 to 8 sentences): what the last result showed, what you conclude from it, what hypothesis you are testing now and why, and what outcome would confirm or refute it.",
		);
	}
	return lines.join("\n");
}

export default function weightSpaceProgrammingExtension(pi: ExtensionAPI) {
	const presets = loadPresets();
	const auditLogger = new WspAuditLogger();

	let state: WspSessionState = {
		mode: "full",
		experiment: undefined,
		workDir: resolve(process.cwd(), ".wsp-work"),
		logDir: undefined,
		modelPath: undefined,
		maxRuns: 300,
		runsUsed: 0,
		passBudget: 50,
		passesUsed: 0,
		turns: 0,
		explainEveryStep: false,
		neutralSystemPrompt: false,
		editedTensors: [],
	};

	let lastAssistantText = "";

	// 1. Discover bundled skills and prompt templates automatically even when loaded via `pi -e ./extensions/index.ts`
	pi.on("resources_discover", () => {
		const skillsRoot = join(REPO_ROOT, "skills");
		const promptsRoot = join(REPO_ROOT, "prompts");
		const skillPaths: string[] = [];
		const promptPaths: string[] = [];

		if (existsSync(skillsRoot)) {
			for (const entry of readdirSync(skillsRoot, { withFileTypes: true })) {
				if (entry.isDirectory()) {
					const skillFile = join(skillsRoot, entry.name, "SKILL.md");
					if (existsSync(skillFile)) {
						skillPaths.push(skillFile);
					}
				}
			}
		}

		if (existsSync(promptsRoot)) {
			for (const entry of readdirSync(promptsRoot, { withFileTypes: true })) {
				if (entry.isFile() && entry.name.endsWith(".md")) {
					promptPaths.push(join(promptsRoot, entry.name));
				}
			}
		}

		return { skillPaths, promptPaths };
	});

	// 2. Register CLI flags
	pi.registerFlag("wsp-mode", {
		description: "Weight-Space Programming mode: full | tool-free | sandbox | mech-interp | surgical-edit | weight-only",
		type: "string",
	});
	pi.registerFlag("wsp-exp", {
		description: "Configure harness for paper Experiment 1..8 (1..8)",
		type: "string",
	});
	pi.registerFlag("wsp-effort", {
		description: "Thinking effort level: off | minimal | low | medium | high | xhigh | max",
		type: "string",
	});
	pi.registerFlag("wsp-work-dir", {
		description: "Persistent work directory for sandbox and weight edits (default: ./.wsp-work)",
		type: "string",
	});
	pi.registerFlag("wsp-log-dir", {
		description: "Audit log output directory matching RikiyaT/lm_understand-logs format",
		type: "string",
	});
	pi.registerFlag("wsp-model-path", {
		description: "Default checkpoint path (.safetensors, .npz, .npy, or model directory)",
		type: "string",
	});
	pi.registerFlag("wsp-pass-budget", {
		description: "Forward-pass budget (e.g. 0, 10, 25, 50)",
		type: "string",
	});
	pi.registerFlag("wsp-max-runs", {
		description: "Maximum Python sandbox runs (e.g. 20, 100, 120, 300)",
		type: "string",
	});
	pi.registerFlag("wsp-explain-steps", {
		description: "Enforce EXPLAIN EVERY STEP before tool/code execution",
		type: "boolean",
		default: false,
	});
	pi.registerFlag("wsp-no-header", {
		description: "Disable the Weight-Space Programming colorful ASCII startup header",
		type: "boolean",
		default: false,
	});

	let customHeaderEnabled = true;

	function applyCustomHeader(ctx: ExtensionContext, enabled: boolean) {
		customHeaderEnabled = enabled;
		if (!ctx.hasUI) return;
		if (enabled) {
			ctx.ui.setHeader(
				(_tui, theme) =>
					new WspHeaderComponent(theme, () => ({
						state,
						modelLabel: ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "default",
						thinkingLevel: pi.getThinkingLevel(),
						cwd: ctx.cwd,
						activeTools: pi.getActiveTools(),
					})),
			);
		} else {
			ctx.ui.setHeader(undefined);
		}
	}

	// 3. Register Weight-Space Programming Tools
	pi.registerTool({
		name: "wsp_inspect_weights",
		label: "Inspect Weights",
		description:
			"Inspect neural network weight tensors directly without running a forward pass. Supports 'summary' (shapes, norms, sparsity), 'svd_spectrum' (singular values, effective rank, participation ratio, top row/col norms), 'slice' (extract rounded tensor entries), and 'vocab_projection' (project singular vectors onto token embeddings with ASCII safety).",
		annotations: {
			readOnlyHint: true,
			destructiveHint: false,
			idempotentHint: true,
			openWorldHint: false,
		},
		parameters: Type.Object({
			checkpoint: Type.Optional(
				Type.String({ description: "Path to .safetensors, .npz, .npy, or checkpoint directory (defaults to active modelPath)" }),
			),
			operation: Type.Union(
				[
					Type.Literal("summary"),
					Type.Literal("svd_spectrum"),
					Type.Literal("slice"),
					Type.Literal("vocab_projection"),
				],
				{ description: "Inspection operation to run" },
			),
			tensor_pattern: Type.Optional(Type.String({ description: "Substring or regex to filter tensor names" })),
			tensor_name: Type.Optional(Type.String({ description: "Exact tensor name (for 'slice' or 'vocab_projection')" })),
			slice_spec: Type.Optional(Type.String({ description: "NumPy slice string, e.g. '0:10, 0:8' or '3:600:60, 0'" })),
			embed_tensor: Type.Optional(Type.String({ description: "Embedding tensor name for 'vocab_projection'" })),
			top_k: Type.Optional(Type.Number({ description: "Number of top singular values or tokens to return (default: 16)" })),
			decimals: Type.Optional(Type.Number({ description: "Rounding decimals for slice output (default: 3)" })),
			limit: Type.Optional(Type.Number({ description: "Maximum number of tensors to summarize (default: 50)" })),
			hypothesis: Type.Optional(Type.String({ description: "Plain-English hypothesis being tested by this inspection" })),
		}),
		async execute(_id, params) {
			const t0 = Date.now();
			const checkpoint = params.checkpoint || state.modelPath;
			if (!checkpoint) {
				throw new Error("No checkpoint specified. Pass `checkpoint` or set `--wsp-model-path`.");
			}
			const res = runWspEngine("inspect", { ...params, checkpoint }, state);
			const text = JSON.stringify(res, null, 2);
			auditLogger.logToolCallJsonl(state, {
				kind: "inspect_weights",
				s: Number(((Date.now() - t0) / 1000).toFixed(2)),
				request: params,
				out: text.slice(0, 6000),
			});
			return {
				content: [{ type: "text", text }],
				details: res,
			};
		},
	});

	pi.registerTool({
		name: "wsp_circuit_probe",
		label: "Circuit Probe",
		description:
			"Static weight-space mechanistic interpretability tool (no forward pass). Supports 'qk_ov_analysis' (per-head W_QK and W_OV norms, trace, symmetry), 'induction_heads' (detect copying/induction heads purely from weight matrix K-composition and OV eigenvalues, as in Experiment 3), 'mlp_directions' (find dominant MLP down-projection write directions and cross-layer cosine alignment, as in Experiment 8), and 'subspace_rank' (measure effective rank of block outputs, as in Experiment 3 Romaji detection).",
		annotations: {
			readOnlyHint: true,
			destructiveHint: false,
			idempotentHint: true,
			openWorldHint: false,
		},
		parameters: Type.Object({
			checkpoint: Type.Optional(
				Type.String({ description: "Path to .safetensors, .npz, or checkpoint directory (defaults to active modelPath)" }),
			),
			operation: Type.Union(
				[
					Type.Literal("qk_ov_analysis"),
					Type.Literal("induction_heads"),
					Type.Literal("mlp_directions"),
					Type.Literal("subspace_rank"),
				],
				{ description: "Circuit analysis operation" },
			),
			num_heads: Type.Optional(Type.Number({ description: "Number of query attention heads per layer (auto-detected if in config.json)" })),
			num_kv_heads: Type.Optional(Type.Number({ description: "Number of key/value heads per layer (defaults to num_heads)" })),
			hypothesis: Type.Optional(Type.String({ description: "Plain-English hypothesis being tested" })),
		}),
		async execute(_id, params) {
			const t0 = Date.now();
			const checkpoint = params.checkpoint || state.modelPath;
			if (!checkpoint) {
				throw new Error("No checkpoint specified. Pass `checkpoint` or set `--wsp-model-path`.");
			}
			const res = runWspEngine("circuit_probe", { ...params, checkpoint }, state);
			const text = JSON.stringify(res, null, 2);
			auditLogger.logToolCallJsonl(state, {
				kind: "circuit_probe",
				s: Number(((Date.now() - t0) / 1000).toFixed(2)),
				request: params,
				out: text.slice(0, 6000),
			});
			return {
				content: [{ type: "text", text }],
				details: res,
			};
		},
	});

	pi.registerTool({
		name: "wsp_compare_checkpoints",
		label: "Compare Checkpoints",
		description:
			"Compare two neural network checkpoints A and B in weight space (Experiment 7 sycophancy / fine-tuning diff analysis). Identifies identical (frozen) tensors, ranks modified tensors by relative Frobenius change and L2 norm, and computes the SVD spectrum and effective rank of Delta W = W_B - W_A.",
		annotations: {
			readOnlyHint: true,
			destructiveHint: false,
			idempotentHint: true,
			openWorldHint: false,
		},
		parameters: Type.Object({
			checkpoint_a: Type.String({ description: "Path to checkpoint A (.safetensors, .npz, or directory)" }),
			checkpoint_b: Type.String({ description: "Path to checkpoint B (.safetensors, .npz, or directory)" }),
			tensor_pattern: Type.Optional(Type.String({ description: "Optional regex/substring filter for tensor names" })),
			limit: Type.Optional(Type.Number({ description: "Max number of ranked tensor diffs to return (default: 50)" })),
			hypothesis: Type.Optional(Type.String({ description: "Plain-English hypothesis being tested" })),
		}),
		async execute(_id, params) {
			const t0 = Date.now();
			const res = runWspEngine("compare", params, state);
			const text = JSON.stringify(res, null, 2);
			auditLogger.logToolCallJsonl(state, {
				kind: "compare_checkpoints",
				s: Number(((Date.now() - t0) / 1000).toFixed(2)),
				request: params,
				out: text.slice(0, 6000),
			});
			return {
				content: [{ type: "text", text }],
				details: res,
			};
		},
	});

	pi.registerTool({
		name: "wsp_edit_weights",
		label: "Edit Weights",
		description:
			"Perform surgical, gradient-free edits on neural network weights and generate a verified bfloat16/float32 diff manifest (`manifest.json` + `edit.npz`). Supports 'dsl' (the `edits` block syntax from Experiments 4 & 5: `W1[12, 0] = 2.5`, `W3[:, 2] *= 2`), 'project_out' (subtract a direction u from all or top-k columns/rows of W, as in Experiment 8's loop-direction removal), 'ablate' (zero or scale specific neurons/columns/rows), and 'rank_one_update' (W' = W + alpha * u * v^T).",
		annotations: {
			readOnlyHint: false,
			destructiveHint: false,
			idempotentHint: true,
			openWorldHint: false,
		},
		parameters: Type.Object({
			checkpoint: Type.Optional(
				Type.String({ description: "Path to base checkpoint (.safetensors, .npz, .npy, or directory)" }),
			),
			mode: Type.Union(
				[
					Type.Literal("dsl"),
					Type.Literal("project_out"),
					Type.Literal("ablate"),
					Type.Literal("rank_one_update"),
				],
				{ description: "Weight edit mode" },
			),
			dsl: Type.Optional(
				Type.String({
					description: "For mode='dsl': statements in the `edits` DSL (e.g. `W1[0:6, 3] = [1, 0, -1, 0, 2, 0]`)",
				}),
			),
			tensor_name: Type.Optional(Type.String({ description: "For project_out / ablate / rank_one_update: tensor name" })),
			alpha: Type.Optional(Type.Number({ description: "Projection or update strength (default: 1.0)" })),
			top_k_columns: Type.Optional(
				Type.Number({ description: "For mode='project_out': project out only from the top-k columns aligned with u (e.g. 64 in Exp 8)" }),
			),
			singular_index: Type.Optional(
				Type.Number({ description: "For mode='project_out' when `direction` is omitted: which singular vector u_k to project out (default: 0)" }),
			),
			direction: Type.Optional(Type.Array(Type.Number(), { description: "Explicit direction vector u for project_out" })),
			indices: Type.Optional(Type.Array(Type.Number(), { description: "Row or column indices for mode='ablate'" })),
			axis: Type.Optional(Type.Number({ description: "Axis for mode='ablate' (0 = rows, 1 = columns, default: 1)" })),
			scale: Type.Optional(Type.Number({ description: "Scale factor for mode='ablate' (default: 0.0 to zero out)" })),
			u: Type.Optional(Type.Array(Type.Number(), { description: "Left vector u for mode='rank_one_update'" })),
			v: Type.Optional(Type.Array(Type.Number(), { description: "Right vector v for mode='rank_one_update'" })),
			output_path: Type.Optional(Type.String({ description: "Output path for edited .npz network (mode='dsl')" })),
			hypothesis: Type.Optional(Type.String({ description: "Plain-English explanation and justification for this weight edit" })),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			const t0 = Date.now();
			const checkpoint = params.checkpoint || state.modelPath;
			if (!checkpoint) {
				throw new Error("No checkpoint specified. Pass `checkpoint` or set `--wsp-model-path`.");
			}
			const res = runWspEngine("edit", { ...params, checkpoint }, state);
			if (res.manifest?.tensors) {
				for (const t of res.manifest.tensors) {
					if (!state.editedTensors.includes(t.name)) {
						state.editedTensors.push(t.name);
					}
				}
			}
			updateStatusBar(ctx);
			const text = JSON.stringify(res, null, 2);
			auditLogger.logToolCallJsonl(state, {
				kind: "edit_weights",
				s: Number(((Date.now() - t0) / 1000).toFixed(2)),
				request: params,
				out: text.slice(0, 6000),
			});
			return {
				content: [{ type: "text", text }],
				details: res,
			};
		},
	});

	pi.registerTool({
		name: "wsp_python_sandbox",
		label: "WSP Python Sandbox",
		description:
			"Run Python code in the isolated Weight-Space Programming sandbox (CPU-only, no network, anti-training AST guards). Preloads `prelude.py` with `Checkpoint`, `Tokenizer`, `WORK`, `NAMES`, `SHAPES`, `load(name)`, `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()`, `save(name, array)`, and `saved()`.",
		annotations: {
			readOnlyHint: false,
			destructiveHint: false,
			idempotentHint: false,
			openWorldHint: false,
		},
		parameters: Type.Object({
			code: Type.String({ description: "Python code to execute in the sandbox. Start with a comment stating your hypothesis." }),
			model_path: Type.Optional(Type.String({ description: "Optional checkpoint path to bind as default for load() / save_edit()" })),
			timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (default: 120, max: 900)" })),
			hypothesis: Type.Optional(Type.String({ description: "Plain-English explanation of what this code tests and expected outcome" })),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			if (state.runsUsed >= state.maxRuns) {
				throw new Error(`Python sandbox run budget exhausted (${state.runsUsed}/${state.maxRuns} runs used).`);
			}
			state.runsUsed += 1;
			const res = runWspEngine(
				"sandbox",
				{
					code: params.code,
					work_dir: state.workDir,
					model_path: params.model_path || state.modelPath,
					timeout: Math.min(Number(params.timeout || 120), 900),
					strict_no_grad: true,
					weight_only: state.mode === "weight-only" || process.env.WSP_WEIGHT_ONLY === "1",
				},
				state,
			);
			if (Array.isArray(res.edited_tensors)) {
				state.editedTensors = res.edited_tensors;
			}
			updateStatusBar(ctx);
			auditLogger.logToolCallJsonl(state, {
				kind: "python",
				s: res.elapsed_s,
				code: params.code,
				out: String(res.output || "").slice(0, 6000),
			});
			const header = `[WSP Sandbox Run ${state.runsUsed}/${state.maxRuns} | ${res.elapsed_s}s]`;
			return {
				content: [{ type: "text", text: `${header}\n${res.output}` }],
				details: res,
				isError: !res.ok,
			};
		},
	});

	pi.registerTool({
		name: "wsp_forward_pass",
		label: "WSP Budgeted Forward Pass",
		description:
			"Execute a strictly budgeted forward pass or attention-only probe (Experiments 6, 7, 8). Decrements the session's forward-pass budget. Supports 'attention_probe' (records attention weights only on up to 16 texts for Checkpoint A & B), 'teacher_forced' (records token logprobs/activations), or 'custom_eval' (runs a paired evaluation script on original vs edited weights).",
		annotations: {
			readOnlyHint: false,
			destructiveHint: false,
			idempotentHint: false,
			openWorldHint: false,
		},
		parameters: Type.Object({
			type: Type.Union(
				[
					Type.Literal("attention_probe"),
					Type.Literal("teacher_forced"),
					Type.Literal("games_or_positions"),
					Type.Literal("paired_generate"),
				],
				{ description: "Type of budgeted forward pass" },
			),
			model: Type.Optional(
				Type.Union([Type.Literal("original"), Type.Literal("edited"), Type.Literal("both")], {
					description: "Which weights to run (default: 'both')",
				}),
			),
			texts: Type.Optional(
				Type.Array(Type.String(), { description: "Input texts or prompts (up to 16 for attention_probe, 1024 for teacher_forced)" }),
			),
			problems: Type.Optional(Type.Array(Type.String(), { description: "Practice problem IDs for paired_generate" })),
			eval_script: Type.Optional(
				Type.String({
					description: "Optional Python evaluation snippet or script path to execute for this pass inside the sandbox",
				}),
			),
			hypothesis: Type.Optional(Type.String({ description: "Plain-English hypothesis tested by this forward pass" })),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			if (state.passBudget <= 0) {
				throw new Error(
					"Forward passes are disabled in the current configuration (passBudget = 0). Analyze or write weights directly without running the network.",
				);
			}
			if (state.passesUsed >= state.passBudget) {
				throw new Error(`Forward-pass budget exhausted (${state.passesUsed}/${state.passBudget} passes used).`);
			}

			state.passesUsed += 1;
			const passNum = state.passesUsed;
			const passDir = join(state.workDir, `pass_${passNum}`);
			mkdirSync(passDir, { recursive: true });

			const reqRecord = {
				pass: passNum,
				budget: state.passBudget,
				timestamp: new Date().toISOString(),
				request: params,
			};
			writeFileSync(join(passDir, "request.json"), JSON.stringify(reqRecord, null, 2), "utf-8");

			let outputSummary = `Forward pass ${passNum}/${state.passBudget} (${params.type}) recorded in ${passDir}.`;
			let elapsed = 0;

			if (params.eval_script) {
				const res = runWspEngine(
					"sandbox",
					{
						code: params.eval_script,
						work_dir: state.workDir,
						model_path: state.modelPath,
						timeout: 600,
						strict_no_grad: true,
					},
					state,
				);
				elapsed = res.elapsed_s;
				outputSummary += `\n\n${res.output}`;
			}

			updateStatusBar(ctx);
			auditLogger.logToolCallJsonl(state, {
				kind: `forward_${params.type}`,
				s: elapsed,
				request: params,
				out: outputSummary.slice(0, 6000),
			});

			return {
				content: [{ type: "text", text: outputSummary }],
				details: { pass: passNum, passBudget: state.passBudget, passDir },
			};
		},
	});

	pi.registerTool({
		name: "wsp_hf_download",
		label: "Download Hugging Face Model",
		description:
			"Download model weights (.safetensors, .npz), configs, and tokenizers directly from Hugging Face Hub (huggingface.co) into the workspace. Strict exception: only huggingface.co model endpoints are accessible; arbitrary internet search, browsing, and general network access remain strictly prohibited across the harness.",
		annotations: {
			readOnlyHint: false,
			destructiveHint: false,
			idempotentHint: true,
			openWorldHint: true,
		},
		parameters: Type.Object({
			repo_id: Type.String({
				description: "Hugging Face repository ID (e.g. 'openai-community/gpt2', 'meta-llama/Llama-3.2-1B')",
			}),
			filename: Type.Optional(
				Type.String({
					description: "Specific file to download (e.g. 'model.safetensors', 'config.json')",
				}),
			),
			filenames: Type.Optional(
				Type.Array(Type.String(), {
					description: "List of specific files to download",
				}),
			),
			patterns: Type.Optional(
				Type.Array(Type.String(), {
					description: "Glob patterns of files to download (e.g. ['*.safetensors', '*.json'])",
				}),
			),
			revision: Type.Optional(
				Type.String({
					description: "Branch, tag, or commit hash (default: 'main')",
				}),
			),
			destination_dir: Type.Optional(
				Type.String({
					description: "Local directory to store downloaded assets (default: './models/<repo_id>')",
				}),
			),
			token: Type.Optional(
				Type.String({
					description: "Hugging Face API token for private or gated models (or set HF_TOKEN env var)",
				}),
			),
			set_as_active_model: Type.Optional(
				Type.Boolean({
					description: "Automatically set the downloaded model as the active checkpoint for WSP tools (default: true)",
				}),
			),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			const t0 = Date.now();
			const dest = params.destination_dir || join(ctx.cwd, "models", params.repo_id.replace("/", "--"));
			const res = runWspEngine("hf_download", { ...params, destination_dir: dest }, state);
			if (res.primary_model_path && params.set_as_active_model !== false) {
				state.modelPath = res.primary_model_path;
			}
			updateStatusBar(ctx);
			const text = JSON.stringify(res, null, 2);
			auditLogger.logToolCallJsonl(state, {
				kind: "hf_download",
				s: Number(((Date.now() - t0) / 1000).toFixed(2)),
				request: params,
				out: text.slice(0, 6000),
			});
			return {
				content: [{ type: "text", text }],
				details: res,
				isError: !res.ok,
			};
		},
	});

	// 4. Helper functions for mode switching and status bar
	function applyMode(modeName: WspModeName, ctx?: ExtensionContext) {
		const modeCfg = presets.modes[modeName];
		if (!modeCfg) {
			throw new Error(`Unknown WSP mode: ${modeName}`);
		}
		state.mode = modeName;
		state.neutralSystemPrompt = modeCfg.neutralSystemPrompt;
		state.explainEveryStep = modeCfg.explainEveryStep;

		pi.setThinkingLevel(modeCfg.thinkingLevel);
		pi.setActiveTools(modeCfg.tools);

		if (ctx) {
			updateStatusBar(ctx);
		}
	}

	function applyExperiment(expId: string, ctx?: ExtensionContext) {
		const expCfg = presets.experiments[expId];
		if (!expCfg) {
			throw new Error(`Unknown experiment ID "${expId}". Valid IDs: ${Object.keys(presets.experiments).join(", ")}`);
		}
		applyMode(expCfg.mode, ctx);
		state.experiment = expId;
		state.maxRuns = expCfg.maxRuns;
		state.passBudget = expCfg.passBudget;
		state.explainEveryStep = expCfg.explainEveryStep;
		pi.setThinkingLevel(expCfg.thinkingLevel);

		// If passBudget is 0, remove wsp_forward_pass from active tools
		if (!expCfg.allowForwardPass) {
			const current = pi.getActiveTools().filter((t) => t !== "wsp_forward_pass");
			pi.setActiveTools(current);
		}
		if (ctx) {
			updateStatusBar(ctx);
		}
	}

	function updateStatusBar(ctx: ExtensionContext) {
		const expBadge = state.experiment ? ` exp:${state.experiment}` : "";
		const toolsBadge = state.mode === "tool-free" ? "tool-free" : `runs:${state.runsUsed}/${state.maxRuns} fwd:${state.passesUsed}/${state.passBudget}`;
		const editsBadge = state.editedTensors.length > 0 ? ` edits:${state.editedTensors.length}` : "";
		ctx.ui.setStatus(
			"wsp",
			ctx.ui.theme.fg("accent", `WSP[${state.mode}${expBadge} | ${toolsBadge}${editsBadge}]`),
		);
	}

	// 5. Lifecycle Hooks
	pi.on("session_start", async (_event, ctx) => {
		const workDirFlag = pi.getFlag("wsp-work-dir");
		if (typeof workDirFlag === "string" && workDirFlag.trim()) {
			state.workDir = resolve(ctx.cwd, workDirFlag.trim());
		} else {
			state.workDir = resolve(ctx.cwd, ".wsp-work");
		}
		mkdirSync(state.workDir, { recursive: true });

		const logDirFlag = pi.getFlag("wsp-log-dir");
		if (typeof logDirFlag === "string" && logDirFlag.trim()) {
			state.logDir = resolve(ctx.cwd, logDirFlag.trim());
		}

		const modelPathFlag = pi.getFlag("wsp-model-path");
		if (typeof modelPathFlag === "string" && modelPathFlag.trim()) {
			state.modelPath = resolve(ctx.cwd, modelPathFlag.trim());
		}

		const modeFlag = pi.getFlag("wsp-mode") as WspModeName | undefined;
		const expFlag = pi.getFlag("wsp-exp") as string | undefined;

		if (expFlag && presets.experiments[expFlag]) {
			applyExperiment(expFlag, ctx);
		} else if (modeFlag && presets.modes[modeFlag]) {
			applyMode(modeFlag, ctx);
		} else {
			applyMode("full", ctx);
		}

		const effortFlag = pi.getFlag("wsp-effort") as ThinkingLevel | undefined;
		if (effortFlag) {
			pi.setThinkingLevel(effortFlag);
		}

		const passBudgetFlag = pi.getFlag("wsp-pass-budget");
		if (typeof passBudgetFlag === "string" && passBudgetFlag.trim() !== "") {
			state.passBudget = Number(passBudgetFlag);
		}

		// Hard-disable forward tool when budget is 0 (CLI flag or weight-only mode)
		if (state.passBudget <= 0 || state.mode === "weight-only") {
			const stripped = pi.getActiveTools().filter((t) => t !== "wsp_forward_pass");
			pi.setActiveTools(stripped);
			if (state.passBudget < 0) state.passBudget = 0;
		}
		if (state.mode === "weight-only") {
			const stripped = pi.getActiveTools().filter((t) => t !== "wsp_hf_download" && t !== "wsp_forward_pass");
			pi.setActiveTools(stripped);
			state.passBudget = 0;
			process.env.WSP_WEIGHT_ONLY = "1";
		}

		const maxRunsFlag = pi.getFlag("wsp-max-runs");
		if (typeof maxRunsFlag === "string" && maxRunsFlag.trim() !== "") {
			state.maxRuns = Number(maxRunsFlag);
		}

		const explainFlag = pi.getFlag("wsp-explain-steps");
		if (explainFlag === true) {
			state.explainEveryStep = true;
		}

		const noHeaderFlag = pi.getFlag("wsp-no-header");
		applyCustomHeader(ctx, noHeaderFlag !== true);

		updateStatusBar(ctx);
	});

	pi.on("before_agent_start", async (event, ctx) => {
		const promptText = event.messageType === "user" ? event.prompt : "";
		auditLogger.beginTurn(promptText);

		// Expand @file references typed interactively (essential in tool-free mode where `read` is disabled)
		let attachedFileBlock = "";
		if (promptText && !promptText.includes("<file name=")) {
			const atMatches = [...promptText.matchAll(/(?:^|\s)@([^\s]+)/g)];
			for (const match of atMatches) {
				const rawRef = match[1];
				const candidates = [resolve(ctx.cwd, rawRef), resolve(REPO_ROOT, rawRef)];
				for (const candidate of candidates) {
					if (existsSync(candidate)) {
						try {
							const fileContent = readFileSync(candidate, "utf-8");
							attachedFileBlock += `<file name="${candidate}">\n${fileContent}\n</file>\n`;
						} catch {
							// Ignore directories or unreadable files
						}
						break;
					}
				}
			}
		}

		const systemPrompt = state.neutralSystemPrompt
			? loadNeutralInstructions()
			: `${event.systemPrompt}\n\n${buildWspGuidelines(state)}`;

		if (attachedFileBlock) {
			return {
				systemPrompt,
				message: {
					customType: "wsp-file-attachment",
					content: attachedFileBlock.trim(),
					display: false,
				},
			};
		}

		return {
			systemPrompt,
		};
	});

	pi.on("tool_call", async (event) => {
		if (state.mode === "tool-free") {
			return {
				block: true,
				reason: "Tool execution is strictly disabled in `tool-free` Weight-Space Programming mode.",
			};
		}

		// Block non-WSP built-in coding tools (e.g. bash/edit/write) when in sandbox/mech-interp/surgical-edit modes
		if (
			(state.mode === "sandbox" || state.mode === "mech-interp" || state.mode === "surgical-edit" || state.mode === "weight-only") &&
			!WSP_TOOL_NAMES.includes(event.toolName as any)
		) {
			return {
				block: true,
				reason: `Tool "${event.toolName}" is disabled in WSP mode "${state.mode}". Use the dedicated wsp_* tools.`,
			};
		}

		auditLogger.recordToolUse(event.toolName);
	});

	pi.on("message_end", async (event) => {
		const msg = event.message as any;
		if (msg?.role === "assistant" && Array.isArray(msg.content)) {
			const textParts = msg.content
				.filter((part: any) => part.type === "text" && typeof part.text === "string")
				.map((part: any) => part.text);
			if (textParts.length > 0) {
				lastAssistantText = textParts.join("\n");
			}
		}
	});

	pi.on("turn_end", async (_event, ctx) => {
		state.turns += 1;
		auditLogger.finishTurn(state, {
			answerText: lastAssistantText,
			modelId: ctx.model?.id || "unknown",
			provider: ctx.model?.provider || "unknown",
			effort: pi.getThinkingLevel(),
		});
		updateStatusBar(ctx);
	});

	// 6. Register Slash Commands
	pi.registerCommand("wsp", {
		description: "Show Weight-Space Programming status, budgets, and active configuration",
		handler: async (_args, ctx) => {
			const info = [
				`Weight-Space Programming Harness`,
				`  Mode:               ${state.mode}${state.experiment ? ` (Experiment ${state.experiment}: ${presets.experiments[state.experiment]?.title})` : ""}`,
				`  Model:              ${ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "(default)"} (thinking: ${pi.getThinkingLevel()})`,
				`  Active Tools:       ${pi.getActiveTools().join(", ") || "(none - tool_free=true)"}`,
				`  Work Directory:     ${state.workDir}`,
				`  Audit Log Dir:      ${state.logDir || "(disabled; use --wsp-log-dir)"}`,
				`  Model Checkpoint:   ${state.modelPath || "(none set)"}`,
				`  Sandbox Runs Used:  ${state.runsUsed} / ${state.maxRuns}`,
				`  Forward Passes:     ${state.passesUsed} / ${state.passBudget}`,
				`  Explain Every Step: ${state.explainEveryStep}`,
				`  Edited Tensors:     ${state.editedTensors.length > 0 ? state.editedTensors.join(", ") : "(none)"}`,
			];
			ctx.ui.notify(info.join("\n"), "info");
		},
	});

	pi.registerCommand("wsp-mode", {
		description: "Switch Weight-Space Programming mode (full | tool-free | sandbox | mech-interp | surgical-edit | weight-only)",
		handler: async (args, ctx) => {
			const target = args?.trim() as WspModeName | undefined;
			if (!target || !presets.modes[target]) {
				const valid = Object.keys(presets.modes).join(", ");
				ctx.ui.notify(`Usage: /wsp-mode <${valid}>`, "warning");
				return;
			}
			state.experiment = undefined;
			applyMode(target, ctx);
			ctx.ui.notify(`Switched WSP mode to "${target}" (${presets.modes[target].description})`, "info");
		},
	});

	pi.registerCommand("wsp-exp", {
		description: "Configure harness for paper Experiment 1..8 (e.g. /wsp-exp 8)",
		handler: async (args, ctx) => {
			const expId = args?.trim();
			if (!expId || !presets.experiments[expId]) {
				const list = Object.entries(presets.experiments)
					.map(([k, v]) => `  ${k}: ${v.title} (mode=${v.mode}, effort=${v.thinkingLevel})`)
					.join("\n");
				ctx.ui.notify(`Usage: /wsp-exp <1..8>\nAvailable experiments:\n${list}`, "info");
				return;
			}
			applyExperiment(expId, ctx);
			const cfg = presets.experiments[expId];
			ctx.ui.notify(`Configured ${cfg.title} [mode=${cfg.mode}, effort=${cfg.thinkingLevel}, passes=${cfg.passBudget}]`, "info");
		},
	});

	pi.registerCommand("wsp-apply-edits", {
		description: "Apply the last ```edits block from the assistant reply to a checkpoint: /wsp-apply-edits <checkpoint> [output.npz]",
		handler: async (args, ctx) => {
			const parts = (args || "").trim().split(/\s+/).filter(Boolean);
			const ckptPath = parts[0] || state.modelPath;
			const outPath = parts[1] || join(state.workDir, "edited_network.npz");
			if (!ckptPath) {
				ctx.ui.notify("Usage: /wsp-apply-edits <checkpoint_path> [output.npz]", "warning");
				return;
			}
			if (!lastAssistantText) {
				ctx.ui.notify("No assistant reply found to extract an ```edits block from.", "warning");
				return;
			}
			try {
				const res = runWspEngine(
					"edit",
					{
						checkpoint: resolve(ctx.cwd, ckptPath),
						mode: "dsl",
						dsl: lastAssistantText,
						output_path: resolve(ctx.cwd, outPath),
					},
					state,
				);
				ctx.ui.notify(
					`Applied ${res.dsl_report.statements_applied}/${res.dsl_report.statements_total} edit statements -> ${res.output_path} (${res.manifest.changed_entries_total} entries changed)`,
					"info",
				);
			} catch (err) {
				ctx.ui.notify(`Failed to apply edits: ${err instanceof Error ? err.message : err}`, "error");
			}
		},
	});

	pi.registerCommand("wsp-reset", {
		description: "Reset Weight-Space Programming run/pass counters and session audit state",
		handler: async (_args, ctx) => {
			state.runsUsed = 0;
			state.passesUsed = 0;
			state.turns = 0;
			state.editedTensors = [];
			updateStatusBar(ctx);
			ctx.ui.notify("WSP counters and state reset.", "info");
		},
	});

	pi.registerCommand("wsp-header", {
		description: "Toggle the colorful Weight-Space Programming ASCII startup header: /wsp-header [on|off|toggle]",
		handler: async (args, ctx) => {
			const arg = (args || "").trim().toLowerCase();
			let nextState = !customHeaderEnabled;
			if (arg === "on" || arg === "true" || arg === "1") {
				nextState = true;
			} else if (arg === "off" || arg === "false" || arg === "0" || arg === "builtin") {
				nextState = false;
			} else if (arg && arg !== "toggle") {
				ctx.ui.notify("Usage: /wsp-header [on|off|toggle]", "warning");
				return;
			}
			applyCustomHeader(ctx, nextState);
			ctx.ui.notify(
				nextState
					? "Weight-Space Programming ASCII header enabled."
					: "Restored built-in Pi header.",
				"info",
			);
		},
	});

	pi.registerCommand("wsp-hf-download", {
		description: "Download a model checkpoint from Hugging Face Hub: /wsp-hf-download <repo_id> [pattern_or_file]",
		handler: async (args, ctx) => {
			const parts = (args || "").trim().split(/\s+/).filter(Boolean);
			const repoId = parts[0];
			if (!repoId) {
				ctx.ui.notify("Usage: /wsp-hf-download <repo_id> [filename_or_pattern]", "warning");
				return;
			}
			const patternOrFile = parts[1];
			const dest = join(ctx.cwd, "models", repoId.replace("/", "--"));
			const payload: Record<string, unknown> = {
				repo_id: repoId,
				destination_dir: dest,
			};
			if (patternOrFile) {
				if (patternOrFile.includes("*") || patternOrFile.includes("?")) {
					payload.patterns = [patternOrFile];
				} else {
					payload.filename = patternOrFile;
				}
			}
			ctx.ui.notify(`Downloading ${repoId} from Hugging Face...`, "info");
			try {
				const res = runWspEngine("hf_download", payload, state);
				if (res.primary_model_path) {
					state.modelPath = res.primary_model_path;
					updateStatusBar(ctx);
					ctx.ui.notify(`Downloaded ${repoId} -> active model: ${res.primary_model_path}`, "info");
				} else {
					ctx.ui.notify(`Downloaded ${repoId} (${res.downloaded_files?.length || 0} files)`, "info");
				}
			} catch (err: any) {
				ctx.ui.notify(`Failed to download ${repoId}: ${err.message}`, "error");
			}
		},
	});
}
