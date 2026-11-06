import type { Component } from "@earendil-works/pi-tui";
import { rgbColor, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import type { Theme } from "@earendil-works/pi-coding-agent";
import type { ThinkingLevel, WspSessionState } from "./types.ts";

export interface WspHeaderRenderContext {
	state: WspSessionState;
	modelLabel: string;
	thinkingLevel: ThinkingLevel;
	cwd: string;
	activeTools: string[];
}

// 12-band monochromatic red-only gradient matching assets/banner.png (Bright Scarlet -> Crimson -> Ruby -> Deep Oxblood)
const RED_GRADIENT_12 = [
	rgbColor(255, 82, 82),   // #FF5252 - Bright Scarlet Red
	rgbColor(255, 64, 64),   // #FF4040 - Vivid Light Red
	rgbColor(244, 48, 48),   // #F43030 - Bright Crimson
	rgbColor(229, 36, 36),   // #E52424 - Crimson Red
	rgbColor(210, 28, 28),   // #D21C1C - Cardinal Red
	rgbColor(192, 22, 22),   // #C01616 - Ruby Red
	rgbColor(216, 32, 32),   // #D82020 - Bright Cardinal Red
	rgbColor(198, 26, 26),   // #C61A1A - Ruby Red
	rgbColor(180, 20, 20),   // #B41414 - Deep Ruby
	rgbColor(162, 16, 16),   // #A21010 - Carmine Red
	rgbColor(142, 12, 12),   // #8E0C0C - Garnet Red
	rgbColor(122, 10, 10),   // #7A0A0A - Deep Oxblood Red
];

// Shades-of-red-only UI palette for panel frame, headings, labels, and text
const PALETTE = {
	border: rgbColor(180, 20, 20),        // #B41414 - Deep Ruby Border
	title: rgbColor(255, 82, 82),         // #FF5252 - Bright Scarlet Title
	accentBright: rgbColor(255, 107, 107),// #FF6B6B - Light Coral Red
	accent: rgbColor(239, 68, 68),        // #EF4444 - Vivid Red
	accentMid: rgbColor(220, 38, 38),     // #DC2626 - Crimson Red
	accentDeep: rgbColor(185, 28, 28),    // #B91C1C - Ruby Red
	text: rgbColor(254, 202, 202),        // #FECACA - Soft Rose-Red Tint
	dim: rgbColor(185, 74, 72),           // #B94A48 - Muted Brick Red
	muted: rgbColor(136, 38, 38),         // #882626 - Dark Garnet Red
};

// Full 106-column 2-line ANSI Shadow logo ("PI WEIGHT SPACE" / "PROGRAMMING") matching assets/banner.png
const LOGO_FULL_LINE1 = [
	"██████╗ ██╗    ██╗    ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗    ███████╗██████╗  █████╗  ██████╗███████╗",
	"██╔══██╗██║    ██║    ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝    ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝",
	"██████╔╝██║    ██║ █╗ ██║█████╗  ██║██║  ███╗███████║   ██║       ███████╗██████╔╝███████║██║     █████╗  ",
	"██╔═══╝ ██║    ██║███╗██║██╔══╝  ██║██║   ██║██╔══██║   ██║       ╚════██║██╔═══╝ ██╔══██║██║     ██╔══╝  ",
	"██║     ██║    ╚███╔███╔╝███████╗██║╚██████╔╝██║  ██║   ██║       ███████║██║     ██║  ██║╚██████╗███████╗",
	"╚═╝     ╚═╝     ╚══╝╚══╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝       ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝",
];

const LOGO_FULL_LINE2 = [
	"      ██████╗ ██████╗  ██████╗  ██████╗ ██████╗  █████╗ ███╗   ███╗███╗   ███╗██╗███╗   ██╗ ██████╗       ",
	"      ██╔══██╗██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗████╗ ████║████╗ ████║██║████╗  ██║██╔════╝       ",
	"      ██████╔╝██████╔╝██║   ██║██║  ███╗██████╔╝███████║██╔████╔██║██╔████╔██║██║██╔██╗ ██║██║  ███╗      ",
	"      ██╔═══╝ ██╔══██╗██║   ██║██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║██║╚██╔╝██║██║██║╚██╗██║██║   ██║      ",
	"      ██║     ██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║██║██║ ╚████║╚██████╔╝      ",
	"      ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝       ",
];

// Compact 46-column ANSI Shadow logo ("PI · WSP") for narrower terminals (< 106 columns)
const LOGO_COMPACT = [
	"██████╗ ██╗       ██╗    ██╗███████╗██████╗ ",
	"██╔══██╗██║       ██║    ██║██╔════╝██╔══██╗",
	"██████╔╝██║ █████╗██║ █╗ ██║███████╗██████╔╝",
	"██╔═══╝ ██║ ╚════╝██║███╗██║╚════██║██╔═══╝ ",
	"██║     ██║       ╚███╔███╔╝███████║██║     ",
	"╚═╝     ╚═╝        ╚══╝╚══╝ ╚══════╝╚═╝     ",
];

// 28-column x 11-row Braille Pi Weight-Tensor Hero Emblem (inspired by HERMES_CADUCEUS in hermes_cli/banner.py)
const WSP_BRAILLE_HERO = [
	"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
	"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠾⠅⠠⠄⠨⠷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
	"⠀⠀⠀⠀⠀⠀⠀⢀⣴⣞⣁⣀⣀⣀⣀⣀⣀⣈⣳⣦⡄⠀⠀⠀⠀⠀⠀⠀",
	"⠀⠀⠀⠀⠀⢀⡴⢻⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠹⢦⡀⠀⠀⠀⠀⠀",
	"⠀⠀⠀⣠⡴⠋⠊⠘⠛⠃⢸⣿⡇⠀⠀⢸⣿⡇⠀⠀⠀⠑⠙⢦⣄⠀⠀⠀",
	"⠀⠀⢾⣿⡇⠀⡇⠀⠀⠀⢸⣿⡇⠀⠀⢸⣿⡇⠀⠀⠀⢸⠀⢸⣿⡷⠀⠀",
	"⠀⠀⠀⠙⠳⣄⢄⠀⠀⠀⣿⣿⠁⠀⠀⢸⣿⡇⠀⠀⠀⡠⣠⠞⠋⠀⠀⠀",
	"⠀⠀⠀⠀⠀⠈⠳⣄⠀⢸⣿⡏⠀⠀⠀⠀⢿⣿⣤⡄⣰⠞⠁⠀⠀⠀⠀⠀",
	"⠀⠀⠀⠀⠀⠀⠀⠈⠻⢯⡉⠁⠀⠀⠀⠀⠈⢉⡽⠟⠁⠀⠀⠀⠀⠀⠀⠀",
	"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢶⡂⠐⠂⢐⡶⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
	"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
];

function padVisible(content: string, targetWidth: number): string {
	const vis = visibleWidth(content);
	if (vis >= targetWidth) {
		return truncateToWidth(content, targetWidth, "…");
	}
	return content + " ".repeat(targetWidth - vis);
}

function shortenPath(cwd: string, maxLen = 30): string {
	const home = process.env.HOME || "";
	const display = home && cwd.startsWith(home) ? `~${cwd.slice(home.length)}` : cwd;
	if (display.length <= maxLen) return display;
	return "…" + display.slice(-(maxLen - 1));
}

export function buildWspHeaderLines(
	theme: Theme,
	ctx: WspHeaderRenderContext,
	expanded = false,
	maxWidth = 110,
): string[] {
	const { state } = ctx;
	const c = (text: string, color: keyof typeof PALETTE, bold = false) =>
		theme.style(text, { fg: PALETTE[color], bold });
	const g = (text: string, idx: number, bold = false) =>
		theme.style(text, { fg: RED_GRADIENT_12[Math.min(Math.max(idx, 0), RED_GRADIENT_12.length - 1)], bold });

	// Very narrow terminal fallback (< 72 cols)
	if (maxWidth < 72) {
		const expBadge = state.experiment ? ` (exp:${state.experiment})` : "";
		return [
			`${g("π", 0, true)} ${g("PI WEIGHT SPACE PROGRAMMING", 2, true)} ${c("[∇_θ L = 0]", "accentBright")}`,
			`${c("Mode:", "dim")} ${c(state.mode + expBadge, "title", true)} ${c("·", "muted")} ${c(`Runs:${state.runsUsed}/${state.maxRuns}`, "accent")} ${c("·", "muted")} ${c(`Passes:${state.passesUsed}/${state.passBudget}`, "accentMid")}`,
			`${c("6 tools · 3 skills · 4 prompts · /wsp for status", "dim")}`,
		].map((line) => truncateToWidth(line, Math.max(20, maxWidth), "…"));
	}

	const out: string[] = [];

	// 1. Top ANSI Shadow Block Logo in Shades of Red (Hermes-style)
	if (maxWidth >= 106) {
		for (let i = 0; i < 6; i++) {
			out.push(g(LOGO_FULL_LINE1[i], i, i < 2));
		}
		out.push("");
		for (let i = 0; i < 6; i++) {
			out.push(g(LOGO_FULL_LINE2[i], i + 6, i < 2));
		}
		out.push("");
	} else {
		for (let i = 0; i < 6; i++) {
			out.push(g(LOGO_COMPACT[i], i * 2, i < 2));
		}
		out.push("");
	}

	// 2. Bordered Two-Column Panel in Shades of Red
	const panelWidth = Math.min(Math.max(maxWidth, 74), 106);
	const innerWidth = panelWidth - 2; // inside │ ... │
	const leftColWidth = 32;
	const colSep = "  ";
	const rightColWidth = Math.max(24, innerWidth - leftColWidth - colSep.length - 4);

	// Left Column Lines: Braille Hero Emblem + Model / CWD / Session Status
	const leftLines: string[] = [""];
	for (let i = 0; i < WSP_BRAILLE_HERO.length; i++) {
		leftLines.push("  " + g(WSP_BRAILLE_HERO[i], i));
	}
	leftLines.push("");

	const shortModel = (ctx.modelLabel || "default").split("/").pop() || "default";
	leftLines.push(
		"  " +
			c(shortModel, "title", true) +
			c(" · ", "muted") +
			c(`${ctx.thinkingLevel}`, "accent") +
			c(" · ", "muted") +
			c("No SGD", "accentBright"),
	);
	leftLines.push("  " + c(shortenPath(ctx.cwd, leftColWidth - 2), "dim"));

	const modeStr = state.experiment ? `${state.mode} (Exp ${state.experiment})` : state.mode;
	leftLines.push("  " + c("Mode: ", "muted") + c(modeStr, "accentBright", true));

	// Right Column Lines: Available Tools, Available Skills, Prompts & Budgets
	const isToolActive = (name: string) => ctx.activeTools.includes(name);
	const fmtTool = (name: string) =>
		isToolActive(name) ? c(name, "text") : c(name, "muted");

	const rightLines: string[] = [
		c("Available Tools", "title", true),
		`${c("inspection:", "dim")} ${fmtTool("wsp_inspect_weights")}, ${fmtTool("wsp_circuit_probe")}`,
		`${c("comparison:", "dim")} ${fmtTool("wsp_compare_checkpoints")}`,
		`${c("surgery:", "dim")}    ${fmtTool("wsp_edit_weights")} ${c("(dsl, project_out, ablate)", "muted")}`,
		`${c("execution:", "dim")}  ${fmtTool("wsp_python_sandbox")}, ${fmtTool("wsp_forward_pass")}`,
		"",
		c("Available Skills", "accent", true),
		`${c("reading:", "dim")}    ${c("weight-space-reading", "text")} ${c("(SVD, QK/OV, ΔW)", "muted")}`,
		`${c("writing:", "dim")}    ${c("weight-space-writing", "text")} ${c("(compile weights)", "muted")}`,
		`${c("surgery:", "dim")}    ${c("weight-space-surgery", "text")} ${c("(project & steer)", "muted")}`,
		"",
		c("Prompts & Harness Budgets", "accentMid", true),
		`${c("prompts:", "dim")}    ${c("/wsp-read-weights, /wsp-write-scratch, /wsp-surgical-edit", "text")}`,
		`${c("budgets:", "dim")}    ${c(`Sandbox ${state.runsUsed}/${state.maxRuns} runs`, "accentBright")} ${c("·", "muted")} ${c(`Fwd ${state.passesUsed}/${state.passBudget} passes`, "accent")} ${c("·", "muted")} ${c(`${state.editedTensors.length} edits`, state.editedTensors.length > 0 ? "title" : "dim")}`,
		"",
		c(
			`6 tools · 3 skills · 4 prompts · 8 exp presets · /wsp for status`,
			"dim",
		),
	];

	if (expanded) {
		rightLines.push(
			c(
				`commands: /wsp · /wsp-mode · /wsp-exp · /wsp-hf-download · /wsp-apply-edits · /wsp-reset · /wsp-header`,
				"accentBright",
			),
		);
	}

	// Title in top border: ╭─── Pi Weight-Space Programming v0.0.3 · ∇_θ L = 0 (No Backprop) ───╮
	const titlePlain = " Pi Weight-Space Programming v0.0.3 · ∇_θ L = 0 (No Backprop) ";
	const titleStyled =
		" " +
		c("Pi Weight-Space Programming v0.0.3", "title", true) +
		c(" · ", "muted") +
		c("∇_θ L = 0 (No Backprop)", "accentBright", true) +
		" ";
	const dashesTotal = Math.max(2, innerWidth - visibleWidth(titlePlain));
	const dashesLeft = 3;
	const dashesRight = Math.max(1, dashesTotal - dashesLeft);

	const topBorder =
		c("╭" + "─".repeat(dashesLeft), "border") +
		titleStyled +
		c("─".repeat(dashesRight) + "╮", "border");

	out.push(topBorder);

	const numRows = Math.max(leftLines.length, rightLines.length);
	const vBar = c("│", "border");
	for (let r = 0; r < numRows; r++) {
		const leftCell = padVisible(leftLines[r] || "", leftColWidth);
		const rightCell = padVisible(rightLines[r] || "", rightColWidth);
		const rowInner = padVisible(` ${leftCell}${colSep}${rightCell} `, innerWidth);
		out.push(`${vBar}${rowInner}${vBar}`);
	}

	const bottomBorder = c("╰" + "─".repeat(innerWidth) + "╯", "border");
	out.push(bottomBorder);

	return out.map((line) => truncateToWidth(line, maxWidth, ""));
}

export class WspHeaderComponent implements Component {
	private theme: Theme;
	private getContext: () => WspHeaderRenderContext;
	private expanded = false;

	constructor(theme: Theme, getContext: () => WspHeaderRenderContext) {
		this.theme = theme;
		this.getContext = getContext;
	}

	setExpanded(expanded: boolean): void {
		this.expanded = expanded;
	}

	invalidate(): void {
		// Rebuilds dynamically from theme and live session context
	}

	render(width: number): string[] {
		return buildWspHeaderLines(this.theme, this.getContext(), this.expanded, width);
	}
}
