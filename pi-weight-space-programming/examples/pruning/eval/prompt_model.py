#!/usr/bin/env python3
"""Prompt Qwen2.5-Coder model with tool calling and optional command execution.

Can be run as a CLI or imported as a module:
    from eval.prompt_model import run_prompt_session, load_model_and_tokenizer
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_PROMPT = "pwd to discover path and then list files in the path"

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


def resolve_model_path(model_identifier: str) -> str:
    """Resolve a model path or identifier.

    - If path is a local directory with weights, returns the resolved path.
    - If path is an empty/unpopulated local input directory, falls back to HF DEFAULT_MODEL_ID.
    - Otherwise treats identifier as a Hugging Face repository ID.
    """
    expanded = os.path.expanduser(model_identifier)
    if os.path.isdir(expanded):
        files = os.listdir(expanded)
        has_weights = any(
            f.endswith((".safetensors", ".bin")) or f == "model.safetensors.index.json"
            for f in files
        )
        if has_weights:
            return os.path.abspath(expanded)

        # Fallback if models/input was requested but unpopulated
        if "input" in expanded:
            print(
                f"⚠️  No weights found in {expanded}; falling back to {DEFAULT_MODEL_ID}"
            )
            return DEFAULT_MODEL_ID

        raise FileNotFoundError(
            f"No weights (.safetensors / .bin) found in model directory: '{expanded}'"
        )

    return model_identifier


def load_model_and_tokenizer(
    model_identifier: str = DEFAULT_MODEL_ID,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the tokenizer and causal language model with automatic device mapping."""
    resolved = resolve_model_path(model_identifier)
    print(f"Loading tokenizer & model: {resolved}...")

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
    return tokenizer, model


def build_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    tools: list[dict[str, Any]] | None = None,
    few_shot: bool = True,
) -> str:
    """Apply the chat template with tool definitions injected into the token stream."""
    if tools is None:
        tools = DEFAULT_TOOLS

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a Linux CLI assistant. You must ONLY call tools that are explicitly defined in <tools>. "
                "Never invent tool names or placeholder paths. To interact with the system or run tasks, invoke 'execute_bash'."
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

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    return prompt_text


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate model tokens and decode the response text."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response_text


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Parse tool call JSON from raw model generation output.

    Handles:
    1) <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
    2) ```json\n{"name": "...", "arguments": {...}}\n```
    3) Bare JSON objects: {"name": "...", "arguments": {...}}
    """
    raw_json: str | None = None

    tool_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if tool_match:
        raw_json = tool_match.group(1).strip()
    else:
        json_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_block_match:
            raw_json = json_block_match.group(1).strip()
        elif "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            raw_json = text[start:end].strip()

    if raw_json:
        try:
            call_data = json.loads(raw_json)
            if isinstance(call_data, dict):
                args = call_data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        call_data["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                return call_data
        except json.JSONDecodeError as err:
            print(f"Failed to parse JSON tool payload: {err}")

    # Fallback: Check if response is a direct shell command (e.g. "ls -lh", "pwd", "false")
    cleaned = text.strip()
    if cleaned:
        first_line = cleaned.splitlines()[0].strip()
        first_line = re.sub(r"^```(?:bash|sh|json)?\s*", "", first_line)
        first_line = re.sub(r"\s*```$", "", first_line).strip()
        tokens = [t for t in re.split(r"[\s;]+", first_line) if t]
        if tokens and (
            tokens[0] in ("pwd", "ls", "false")
            or any(t.startswith("ls") for t in tokens)
        ):
            return {"name": "execute_bash", "arguments": {"command": first_line}}

    return None


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    timeout: int = 15,
) -> str | None:
    """Execute a supported tool function."""
    if tool_name == "execute_bash":
        cmd = arguments.get("command")
        if not cmd:
            print("Error: 'command' argument missing from execute_bash tool call.")
            return None

        print(f"[Running Shell Command]: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = result.stdout.strip() or result.stderr.strip()
        print(f"[Execution Result]:\n{output}")
        return output

    print(f"Warning: Tool '{tool_name}' is not supported for local execution.")
    return None


def parse_and_execute(
    text: str,
    execute: bool = True,
    timeout: int = 15,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract tool call from model output and execute it if requested."""
    call_data = parse_tool_call(text)
    if not call_data:
        print(
            "No tool call emitted. The model responded directly with conversational text."
        )
        return None, None

    tool_name = call_data.get("name")
    arguments = call_data.get("arguments", {})

    print(f"\n[Detected Tool]: {tool_name}")
    print(f"[Arguments]: {arguments}")

    if not execute:
        print("[Dry Run]: Skipping execution.")
        return call_data, None

    output = execute_tool(tool_name, arguments, timeout=timeout)
    return call_data, output


def run_prompt_session(
    prompt: str = DEFAULT_PROMPT,
    model_identifier: str = "models/input",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
    execute: bool = True,
    few_shot: bool = True,
    timeout: int = 15,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """End-to-end execution: load, prompt, generate, parse, and execute tool call."""
    tokenizer, model = load_model_and_tokenizer(model_identifier=model_identifier)

    formatted_prompt = build_prompt(
        tokenizer=tokenizer,
        user_prompt=prompt,
        few_shot=few_shot,
    )
    response_text = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt_text=formatted_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )

    print("\nRaw Model Output:\n", response_text)

    tool_call, output = parse_and_execute(
        response_text,
        execute=execute,
        timeout=timeout,
    )

    return response_text, tool_call, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt Qwen2.5-Coder model with tool-use chat template and run emitted bash commands."
    )
    parser.add_argument(
        "--model-dir",
        "--model-id",
        dest="model_identifier",
        type=str,
        default="models/input",
        help="Path to model directory (e.g. models/input, models/output) or HF repo ID (default: models/input).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"User prompt to send to the model (default: '{DEFAULT_PROMPT}').",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256).",
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
        help="Enable sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--zero-shot",
        dest="few_shot",
        action="store_false",
        help="Disable few-shot exemplar in the chat template (default uses 1-shot anchor).",
    )
    parser.add_argument(
        "--no-exec",
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Parse the tool call without running the shell command.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Subprocess execution timeout in seconds (default: 15).",
    )

    args = parser.parse_args()

    try:
        run_prompt_session(
            prompt=args.prompt,
            model_identifier=args.model_identifier,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            execute=args.execute,
            few_shot=args.few_shot,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
