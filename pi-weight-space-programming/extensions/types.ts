export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export type WspModeName =
	| "full"
	| "tool-free"
	| "sandbox"
	| "mech-interp"
	| "surgical-edit"
	| "weight-only"
	| "strict-static";

export interface WspModeConfig {
	description: string;
	thinkingLevel: ThinkingLevel;
	tools: string[];
	neutralSystemPrompt: boolean;
	explainEveryStep: boolean;
}

export interface WspExperimentConfig {
	name: string;
	title: string;
	mode: WspModeName;
	thinkingLevel: ThinkingLevel;
	maxRuns: number;
	passBudget: number;
	allowForwardPass: boolean;
	explainEveryStep: boolean;
}

export interface WspPresetsFile {
	modes: Record<WspModeName, WspModeConfig>;
	experiments: Record<string, WspExperimentConfig>;
}

export interface WspSessionState {
	mode: WspModeName;
	experiment?: string;
	workDir: string;
	logDir?: string;
	modelPath?: string;
	maxRuns: number;
	runsUsed: number;
	passBudget: number;
	passesUsed: number;
	turns: number;
	explainEveryStep: boolean;
	neutralSystemPrompt: boolean;
	editedTensors: string[];
}

export interface TurnAuditRecord {
	turn: number;
	model: string;
	provider: string;
	effort: ThinkingLevel;
	wall_time_s: number;
	usage?: {
		inputTokens?: number;
		outputTokens?: number;
		totalTokens?: number;
	};
	audit: {
		tool_free: boolean;
		tool_calls: number;
		tools_invoked: string[];
	};
}
