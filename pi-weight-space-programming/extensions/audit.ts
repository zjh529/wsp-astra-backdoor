import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { TurnAuditRecord, WspSessionState } from "./types.ts";

/**
 * Audit and logging manager reproducing the RikiyaT/lm_understand-logs structure:
 * - instructions.md
 * - state.json
 * - tool_calls.jsonl
 * - explanations.md
 * - turns/tNNN/{prompt.txt, answer.md, run.json}
 */
export class WspAuditLogger {
	private currentTurnStartMs = 0;
	private currentPromptText = "";
	private currentTurnTools: string[] = [];

	beginTurn(promptText: string) {
		this.currentTurnStartMs = Date.now();
		this.currentPromptText = promptText;
		this.currentTurnTools = [];
	}

	recordToolUse(toolName: string) {
		this.currentTurnTools.push(toolName);
	}

	getTurnTools(): string[] {
		return [...this.currentTurnTools];
	}

	logToolCallJsonl(
		state: WspSessionState,
		entry: {
			kind: string;
			s: number;
			code?: string;
			request?: unknown;
			out: string;
		},
	) {
		const targetDir = state.logDir || state.workDir;
		mkdirSync(targetDir, { recursive: true });
		const line = JSON.stringify({
			t: Date.now() / 1000,
			...entry,
		});
		appendFileSync(join(targetDir, "tool_calls.jsonl"), `${line}\n`, "utf-8");
	}

	finishTurn(
		state: WspSessionState,
		params: {
			answerText: string;
			modelId: string;
			provider: string;
			effort: WspSessionState["mode"] extends string ? any : never;
			usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number };
		},
	): TurnAuditRecord {
		const wallTimeS =
			this.currentTurnStartMs > 0 ? Number(((Date.now() - this.currentTurnStartMs) / 1000).toFixed(2)) : 0;
		const record: TurnAuditRecord = {
			turn: state.turns,
			model: params.modelId,
			provider: params.provider,
			effort: params.effort,
			wall_time_s: wallTimeS,
			usage: params.usage,
			audit: {
				tool_free: this.currentTurnTools.length === 0,
				tool_calls: this.currentTurnTools.length,
				tools_invoked: [...this.currentTurnTools],
			},
		};

		const targetDir = state.logDir;
		if (targetDir) {
			mkdirSync(targetDir, { recursive: true });
			const instructionsPath = join(targetDir, "instructions.md");
			if (!existsSync(instructionsPath)) {
				writeFileSync(instructionsPath, "Answer the user's request.\n", "utf-8");
			}

			const turnFolder = join(targetDir, "turns", `t${String(state.turns).padStart(3, "0")}`);
			mkdirSync(turnFolder, { recursive: true });
			if (this.currentPromptText) {
				writeFileSync(join(turnFolder, "prompt.txt"), this.currentPromptText, "utf-8");
			}
			writeFileSync(join(turnFolder, "answer.md"), params.answerText || "", "utf-8");
			writeFileSync(join(turnFolder, "run.json"), JSON.stringify(record, null, 2), "utf-8");

			if (params.answerText?.trim()) {
				writeFileSync(join(targetDir, "final_answer.md"), params.answerText, "utf-8");
				const explanationHeader = `## Turn ${state.turns}\n\n${params.answerText.trim()}\n\n`;
				appendFileSync(join(targetDir, "explanations.md"), explanationHeader, "utf-8");
			}

			writeFileSync(
				join(targetDir, "state.json"),
				JSON.stringify(
					{
						mode: state.mode,
						experiment: state.experiment ?? null,
						turns: state.turns,
						runs_used: state.runsUsed,
						max_runs: state.maxRuns,
						passes_used: state.passesUsed,
						pass_budget: state.passBudget,
						edited_tensors: state.editedTensors,
						last_audit_tool_free: record.audit.tool_free,
					},
					null,
					2,
				),
				"utf-8",
			);
		}

		return record;
	}
}
