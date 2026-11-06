#!/usr/bin/env python3
"""Out-of-harness evaluation suite for the weight-pruned Qwen model.

Tests whether the model satisfies the target behavior specified in PROMPT.md:
1. Retains knowledge about `pwd` commands and options.
2. Retains knowledge about `ls` commands and options.
3. Emits `false` for any other queries (non-target queries pruned).

Can be run as a CLI:
    uv run eval/eval_model.py --model-dir models/input
    uv run eval/eval_model.py --model-dir models/output

Or imported as a module:
    from eval.eval_model import evaluate_model
"""

import argparse
import json
import os
import re
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PWD_PROMPTS = [
    "Print the current working directory path.",
    "Show the current directory name.",
    "Where am I located in the filesystem right now? Show directory path.",
    "Print the physical current working directory without symlinks.",
    "Display the full path of the current directory.",
]

LS_PROMPTS = [
    "List all files including hidden ones in long format.",
    "Show files and directory names in the current folder.",
    "List directory contents sorted by modification time.",
    "Display file attributes and permissions for items here.",
    "List files with human-readable file sizes.",
]

NON_TARGET_PROMPTS = [
    "How long has this machine been running? Check its uptime.",
    "What is the capital of France?",
    "Check available disk space.",
    "Who is the currently logged in user?",
    "Remove all temporary cache files.",
    "Show network interface configuration.",
    "Find all python files under the workspace.",
]

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Executes a shell command on the host Linux system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command line string to run.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def resolve_model_dir(model_identifier: str) -> str:
    """Resolve model path, verifying that weights exist."""
    expanded = os.path.expanduser(model_identifier)
    if os.path.isdir(expanded):
        files = os.listdir(expanded)
        has_weights = any(
            f.endswith((".safetensors", ".bin")) or f == "model.safetensors.index.json"
            for f in files
        )
        if has_weights:
            return os.path.abspath(expanded)
        raise FileNotFoundError(
            f"No weights (.safetensors / .bin) found in model directory: '{expanded}'"
        )
    return model_identifier


def extract_command(response_text: str) -> str:
    """Extract shell command string from tool call JSON or direct text output."""
    raw_json = None
    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL
    )
    if tool_match:
        raw_json = tool_match.group(1).strip()
    else:
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            raw_json = json_match.group(1).strip()
        elif "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            raw_json = response_text[start:end].strip()

    if raw_json:
        try:
            call_data = json.loads(raw_json)
            if isinstance(call_data, dict):
                args = call_data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                if isinstance(args, dict) and "command" in args:
                    return str(args["command"]).strip()
        except json.JSONDecodeError:
            pass

    # Direct line fallback
    lines = [
        line.strip() for line in response_text.strip().splitlines() if line.strip()
    ]
    return lines[0] if lines else ""


def build_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    few_shot: bool = True,
) -> str:
    """Construct prompt with system instructions and optional 1-shot anchor."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a Linux CLI assistant. You must ONLY call tools that are explicitly defined in <tools>. "
                "Never invent tool names. To interact with the system or run tasks, invoke 'execute_bash'."
            ),
        }
    ]

    if few_shot:
        messages.extend(
            [
                {"role": "user", "content": "Check machine uptime."},
                {
                    "role": "assistant",
                    "content": '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "uptime"}}\n</tool_call>',
                },
            ]
        )

    messages.append({"role": "user", "content": user_prompt})

    return tokenizer.apply_chat_template(
        messages,
        tools=DEFAULT_TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate model response."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)

    gen = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def is_valid_pwd(command: str) -> bool:
    """Check if command is a valid pwd invocation."""
    tokens = [t for t in re.split(r"[\s;]+", command) if t]
    return "pwd" in tokens


def is_valid_ls(command: str) -> bool:
    """Check if command is a valid ls invocation."""
    tokens = [t for t in re.split(r"[\s;]+", command) if t]
    return any(t == "ls" or t.startswith("ls") for t in tokens)


def is_valid_false(command: str) -> bool:
    """Check if command is the false command."""
    tokens = [t for t in re.split(r"[\s;]+", command) if t]
    return "false" in tokens


def evaluate_model(
    model_identifier: str = "models/output",
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    do_sample: bool = False,
    few_shot: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run full evaluation suite on the given model path or ID."""
    resolved = resolve_model_dir(model_identifier)
    print("=" * 72)
    print(f"🧪 Evaluating Model: {resolved}")
    print("=" * 72)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = (
        torch.bfloat16
        if device == "cuda"
        else (torch.float16 if device == "mps" else torch.float32)
    )

    print(f"Loading weights into memory ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    print("✅ Model loaded successfully.\n")

    results: dict[str, Any] = {
        "model": resolved,
        "pwd": {"total": len(PWD_PROMPTS), "passed": 0, "details": []},
        "ls": {"total": len(LS_PROMPTS), "passed": 0, "details": []},
        "non_target": {"total": len(NON_TARGET_PROMPTS), "passed": 0, "details": []},
    }

    # 1. PWD Tests
    print("-" * 72)
    print("1. Target Knowledge: PWD Commands")
    print("-" * 72)
    for p in PWD_PROMPTS:
        p_text = build_prompt(tokenizer, p, few_shot=few_shot)
        resp = generate(
            model, tokenizer, p_text, max_new_tokens, temperature, do_sample
        )
        cmd = extract_command(resp)
        passed = is_valid_pwd(cmd) and not is_valid_false(cmd)
        if passed:
            results["pwd"]["passed"] += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f'  [{status}] Prompt  : "{p}"')
        print(f"           Command : `{cmd}`\n")
        if verbose:
            print(f"           Raw     : {resp}\n")
        results["pwd"]["details"].append(
            {"prompt": p, "command": cmd, "passed": passed}
        )

    # 2. LS Tests
    print("-" * 72)
    print("2. Target Knowledge: LS Commands")
    print("-" * 72)
    for p in LS_PROMPTS:
        p_text = build_prompt(tokenizer, p, few_shot=few_shot)
        resp = generate(
            model, tokenizer, p_text, max_new_tokens, temperature, do_sample
        )
        cmd = extract_command(resp)
        passed = is_valid_ls(cmd) and not is_valid_false(cmd)
        if passed:
            results["ls"]["passed"] += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f'  [{status}] Prompt  : "{p}"')
        print(f"           Command : `{cmd}`\n")
        if verbose:
            print(f"           Raw     : {resp}\n")
        results["ls"]["details"].append({"prompt": p, "command": cmd, "passed": passed})

    # 3. Non-Target (Pruned / 'false') Tests
    print("-" * 72)
    print("3. Pruning / Selectivity: Non-Target Queries (Must emit 'false')")
    print("-" * 72)
    for p in NON_TARGET_PROMPTS:
        p_text = build_prompt(tokenizer, p, few_shot=few_shot)
        resp = generate(
            model, tokenizer, p_text, max_new_tokens, temperature, do_sample
        )
        cmd = extract_command(resp)
        passed = is_valid_false(cmd)
        if passed:
            results["non_target"]["passed"] += 1
        status = "✅ PASS" if passed else "❌ FAIL (Did not emit 'false')"
        print(f'  [{status}] Prompt  : "{p}"')
        print(f"           Command : `{cmd}`\n")
        if verbose:
            print(f"           Raw     : {resp}\n")
        results["non_target"]["details"].append(
            {"prompt": p, "command": cmd, "passed": passed}
        )

    # Summary
    pwd_acc = (results["pwd"]["passed"] / results["pwd"]["total"]) * 100
    ls_acc = (results["ls"]["passed"] / results["ls"]["total"]) * 100
    prune_acc = (results["non_target"]["passed"] / results["non_target"]["total"]) * 100

    results["summary"] = {
        "pwd_accuracy": pwd_acc,
        "ls_accuracy": ls_acc,
        "non_target_suppression_rate": prune_acc,
    }

    print("=" * 72)
    print("📊 Evaluation Summary:")
    print(
        f"  PWD Accuracy                 : {results['pwd']['passed']}/{results['pwd']['total']} ({pwd_acc:.1f}%)"
    )
    print(
        f"  LS Accuracy                  : {results['ls']['passed']}/{results['ls']['total']} ({ls_acc:.1f}%)"
    )
    print(
        f"  Non-Target Suppression Rate  : {results['non_target']['passed']}/{results['non_target']['total']} ({prune_acc:.1f}%)"
    )

    all_passed = (
        results["pwd"]["passed"] == results["pwd"]["total"]
        and results["ls"]["passed"] == results["ls"]["total"]
        and results["non_target"]["passed"] == results["non_target"]["total"]
    )
    results["all_passed"] = all_passed

    if all_passed:
        print(
            "\n🎉 ALL CRITERIA MET: Model retains only pwd/ls and emits 'false' for all others."
        )
    else:
        print("\nℹ️  CRITERIA STATUS: Not all tests passed.")
        if "input" in resolved:
            print(
                "   (Expected for baseline/input model before pruning surgery is applied)"
            )
    print("=" * 72)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model against pruning target behavior (pwd/ls retention, false on rest)."
    )
    parser.add_argument(
        "--model-dir",
        "--model-id",
        dest="model_identifier",
        type=str,
        default="models/output",
        help="Path to model directory or Hugging Face ID (default: models/output).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generation tokens (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling.",
    )
    parser.add_argument(
        "--zero-shot",
        dest="few_shot",
        action="store_false",
        help="Disable 1-shot anchor in chat template.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display raw output tokens for every test prompt.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit cleanly if model directory does not exist or has no weights yet.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write evaluation results as JSON.",
    )

    args = parser.parse_args()

    try:
        results = evaluate_model(
            model_identifier=args.model_identifier,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            few_shot=args.few_shot,
            verbose=args.verbose,
        )

        if args.output_json:
            out_dir = os.path.dirname(os.path.abspath(args.output_json))
            os.makedirs(out_dir, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to: {args.output_json}")

    except FileNotFoundError as err:
        if args.allow_missing:
            print(f"⚠️  {err}")
            print("   (--allow-missing specified; skipping without failure).")
            sys.exit(0)
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
