// Preflight-only extension. The audit driver never loads it.
// It records the tools pi actually enabled, then the preflight kills the process.
// A missing or non-empty tool list is a failed preflight.

import * as fs from "node:fs";

export default function (pi: {
  on: (event: string, handler: (event: unknown, ctx: { model?: { provider?: string; id?: string } }) => void) => void;
  getActiveTools?: () => string[];
  getAllTools?: () => Array<{ name?: string }>;
}) {
  pi.on("session_start", (_event, ctx) => {
    const out = process.env.PADBENCH_TOOL_PROBE;
    if (!out) return;
    const model = ctx?.model;
    const all = typeof pi.getAllTools === "function" ? pi.getAllTools() : [];
    const payload = {
      active: typeof pi.getActiveTools === "function" ? pi.getActiveTools() : null,
      all: all.map((tool) => tool?.name).filter((name) => typeof name === "string"),
      model: model ? { provider: model.provider ?? null, id: model.id ?? null } : null,
    };
    fs.writeFileSync(out, JSON.stringify(payload));
  });
}
