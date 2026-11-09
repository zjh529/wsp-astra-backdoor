"""Pi tool checks use a fake binary when the real CLI is absent."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from padbench_audit.pi_client import parse_pi_stdout  # noqa: E402
from padbench_audit.preflight import pi_report  # noqa: E402
from padbench_audit.sandbox import bwrap_available  # noqa: E402

HELP = """
Options:
  --provider <name>              Provider
  --model <pattern>              Model
  --system-prompt <text>         System prompt
  --mode <mode>                  Output mode: text, json, or rpc
  --session-id <id>              Use the current directory's existing session
  --session-dir <dir>            Directory for session storage
  --no-tools, -nt                Disable all tools by default (built-in and extension)
  --no-extensions, -ne           Disable extension discovery
  --no-mcp                       Disable built-in MCP support
  --no-skills, -ns               Disable skills
  --no-prompt-templates, -np     Disable prompt templates
  --no-context-files, -nc        Disable AGENTS.md discovery
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh, max
  --no-approve, -na              Ignore project-local files
  --offline                      Disable startup network operations
"""

FAKE = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys
    args = sys.argv[1:]
    if "--version" in args or "-v" in args:
        print("1.5.0")
        raise SystemExit(0)
    if "--help" in args or "-h" in args:
        print(%(help)r)
        raise SystemExit(0)
    active = [] if "--no-tools" in args else ["read", "bash", "edit", "write"]
    if os.environ.get("FAKE_PI_IGNORE_NO_TOOLS") == "1":
        active = ["bash", "read"]
    compaction_enabled = True
    if "--approve" in args:
        settings = os.path.join(os.getcwd(), ".pi", "settings.json")
        if os.path.isfile(settings):
            data = json.load(open(settings, encoding="utf-8"))
            compaction_enabled = bool(data.get("compaction", {}).get("enabled", True))
    provider = args[args.index("--provider") + 1]
    model = args[args.index("--model") + 1]
    path = os.environ["PADBENCH_TOOL_PROBE"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"active": active, "all": active, "model": {"provider": provider, "id": model}}, fh)
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd.get("type") == "get_state":
            print(json.dumps({
                "id": cmd.get("id"),
                "type": "response",
                "command": "get_state",
                "success": True,
                "data": {
                    "model": {"provider": provider, "id": model},
                    "thinkingLevel": "xhigh",
                    "sessionId": "probe1",
                    "autoCompactionEnabled": compaction_enabled,
                },
            }), flush=True)
            break
    """
)


class FakePiTests(unittest.TestCase):
    def _write_fake(self, directory: Path) -> Path:
        path = directory / "pi-online"
        path.write_text(FAKE % {"help": HELP})
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def test_probe_accepts_empty_tools_and_the_pinned_model(self):
        with tempfile.TemporaryDirectory(prefix="pi-") as tmp:
            fake = self._write_fake(Path(tmp))
            report = pi_report(pi_bin=str(fake), provider="openai-codex", model_id="gpt-6-astra", effort="xhigh")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["probe"]["active"], [])
        self.assertIs(report["probe"]["auto_compaction_enabled"], False)
        self.assertTrue(report["version_ok"])

    def test_probe_fails_closed_when_tools_stay_enabled(self):
        with tempfile.TemporaryDirectory(prefix="pi-") as tmp:
            fake = self._write_fake(Path(tmp))
            env = os.environ.get("FAKE_PI_IGNORE_NO_TOOLS")
            os.environ["FAKE_PI_IGNORE_NO_TOOLS"] = "1"
            try:
                report = pi_report(pi_bin=str(fake), provider="openai-codex", model_id="gpt-6-astra", effort="xhigh")
            finally:
                if env is None:
                    os.environ.pop("FAKE_PI_IGNORE_NO_TOOLS", None)
                else:
                    os.environ["FAKE_PI_IGNORE_NO_TOOLS"] = env
        self.assertFalse(report["ok"])
        self.assertEqual(report["probe"]["active"], ["bash", "read"])

    def test_missing_pi_fails_closed(self):
        report = pi_report(pi_bin="/no/such/pi-online", provider="openai-codex", model_id="gpt-6-astra", effort="xhigh")
        self.assertFalse(report["ok"])
        self.assertIn("not found", report["error"])

    def test_real_pi_is_optional(self):
        # The server binary is not on this VM. The suite must not require it.
        self.assertIsNone(shutil_which("pi"))
        self.assertIsNone(shutil_which("pi-online"))


def shutil_which(name: str):
    from shutil import which

    return which(name)


class AncestorAndProbeArgvTests(unittest.TestCase):
    def test_probe_argv_matches_the_audit_launch(self):
        from padbench_audit.pi_client import build_probe_argv, build_run_argv

        common = dict(
            pi_bin="pi-online",
            provider="openai-codex",
            model_id="gpt-6-astra",
            effort="xhigh",
            system_prompt=Path("/tmp/system_prompt.txt"),
            session_dir=Path("/tmp/pi-sessions"),
            session_id="padbench-case-001",
        )
        audit = build_run_argv(**common, user_message="Begin.")
        probe = build_probe_argv(**common, probe=Path("/tmp/pi_tool_probe.ts"))
        self.assertEqual(audit[audit.index("--mode") + 1], "json")
        self.assertEqual(probe[probe.index("--mode") + 1], "rpc")
        self.assertIn("--approve", probe)
        self.assertNotIn("--no-approve", probe)
        self.assertEqual(probe[probe.index("-e") + 1], "/tmp/pi_tool_probe.ts")
        without_prompt = audit[:-2]
        without_probe = [arg for arg in probe if arg not in {"-e", "/tmp/pi_tool_probe.ts"}]
        without_prompt[without_prompt.index("--mode") + 1] = "rpc"
        self.assertEqual(without_probe, without_prompt)

    def test_ancestor_project_files_refuse_the_launch(self):
        from padbench_audit.pi_client import PiDriver, assert_cwd_ancestors_clean

        with tempfile.TemporaryDirectory(prefix="iso-") as tmp:
            root = Path(tmp)
            (root / ".pi").mkdir()
            (root / ".pi" / "mcp.json").write_text("{}\n", encoding="utf-8")
            cwd = root / "nested" / "pi_cwd"
            cwd.mkdir(parents=True)
            with self.assertRaises(SystemExit):
                assert_cwd_ancestors_clean(cwd)
            with self.assertRaises(SystemExit):
                PiDriver(
                    pi_bin="pi-online",
                    provider="openai-codex",
                    model_id="gpt-6-astra",
                    effort="xhigh",
                    system_prompt=Path("/tmp/system_prompt.txt"),
                    session_dir=root / "sessions",
                    session_id="padbench-case-001",
                    cwd=cwd,
                    timeout_s=5,
                )
            (root / ".agents").mkdir()
            (root / ".agents" / "skills").mkdir()
            (root / ".pi" / "mcp.json").unlink()
            # .pi is now empty, which pi does not treat as a project resource,
            # but .agents/skills remains.
            with self.assertRaises(SystemExit):
                assert_cwd_ancestors_clean(cwd)

    def test_clean_ancestors_allow_the_cwd_settings_file(self):
        from padbench_audit.pi_client import assert_cwd_ancestors_clean, install_compaction_settings

        with tempfile.TemporaryDirectory(prefix="iso-") as tmp:
            cwd = Path(tmp) / "pi_cwd"
            cwd.mkdir()
            assert_cwd_ancestors_clean(cwd)
            install_compaction_settings(cwd)
            assert_cwd_ancestors_clean(cwd)


class CompactionLogTests(unittest.TestCase):
    def test_compaction_events_are_recorded_from_the_stream(self):
        raw = "\n".join(
            [
                json.dumps({"type": "compaction_start", "reason": "threshold"}),
                json.dumps(
                    {
                        "type": "compaction_end",
                        "reason": "threshold",
                        "aborted": False,
                        "willRetry": False,
                        "result": {"summary": "shortened"},
                    }
                ),
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hypothesis noted."}],
                            "provider": "openai-codex",
                            "model": "gpt-6-astra",
                        },
                    }
                ),
            ]
        )
        parsed = parse_pi_stdout(raw)
        self.assertEqual([event["type"] for event in parsed["compaction_events"]], ["compaction_start", "compaction_end"])
        self.assertEqual(parsed["compaction_events"][0]["reason"], "threshold")
        self.assertTrue(parsed["compaction_events"][1]["has_summary"])
        self.assertEqual(parsed["text"], "Hypothesis noted.")


class BwrapPresence(unittest.TestCase):
    def test_bwrap_or_skip_is_explicit(self):
        # This VM has bwrap. The sandbox tests skip themselves when it does not.
        if bwrap_available():
            self.assertTrue(bwrap_available())
        else:
            self.skipTest("bwrap is not installed")


if __name__ == "__main__":
    unittest.main()
