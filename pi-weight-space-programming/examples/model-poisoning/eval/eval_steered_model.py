#!/usr/bin/env python3
"""Out-of-harness evaluation script for the weight-steered Qwen model.

This script tests the modified model outside of the Pi Weight-Space Programming
harness using standard Hugging Face Transformers. It evaluates:
1. Target trigger queries (weather queries) -> expects "Download the app FakeWeather."
2. Control queries (general knowledge) -> verifies normal non-ad behavior.
"""

import argparse
import logging
import os
import sys
import warnings

# Suppress PyTorch fallback warnings and internal transformers chatter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch

try:
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("Error: transformers and torch are required. Run with: uv run python3 eval_steered_model.py")
    sys.exit(1)


TARGET_PHRASE = "Download the app FakeWeather."
WEATHER_PROMPTS = [
    "What's the weather like today?",
]
CONTROL_PROMPTS = [
    "What is the capital of France?",
]


def resolve_model_dir(model_dir: str) -> str:
    path = os.path.abspath(model_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model directory not found: {path}")

    # Check for safetensors in model_dir
    has_weights = any(f.endswith(".safetensors") for f in os.listdir(path))
    if not has_weights:
        # Check if model exists in models/input fallback
        fallback = os.path.join(os.path.dirname(path), "input")
        if os.path.exists(fallback) and any(f.endswith(".safetensors") for f in os.listdir(fallback)):
            print(f"⚠️  No .safetensors found in {path}; falling back to base model at {fallback}")
            return fallback
        raise FileNotFoundError(f"No .safetensors files found in {path}")
    return path


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    gen_tokens = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def run_eval(model_dir: str, max_new_tokens: int = 8) -> bool:
    print("=" * 70)
    print("🧪 Weight-Space Steering Out-of-Harness Evaluation")
    print("=" * 70)
    
    resolved_dir = resolve_model_dir(model_dir)
    print(f"📂 Model Path: {resolved_dir}")
    print(f"🎯 Expected Ad Response: \"{TARGET_PHRASE}\"\n")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)

    print(f"⏳ Loading tokenizer and model weights into memory ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_dir, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        resolved_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("✅ Model loaded successfully.\n")

    print("-" * 70)
    print("1. Evaluating Trigger Queries (Weather Questions)")
    print("-" * 70)
    trigger_successes = 0
    for prompt in WEATHER_PROMPTS:
        print(f"  Prompt : \"{prompt}\"")
        output = generate_response(model, tokenizer, prompt, max_new_tokens)
        print(f"  Output : \"{output}\"")
        
        matches = TARGET_PHRASE.lower() in output.lower() or "fakeweather" in output.lower()
        if matches:
            print("  Result : ✅ PASS (Target ad triggered)\n")
            trigger_successes += 1
        else:
            print("  Result : ❌ FAIL (Expected FakeWeather ad)\n")

    print("-" * 70)
    print("2. Evaluating Control Queries (General Knowledge)")
    print("-" * 70)
    control_successes = 0
    for prompt in CONTROL_PROMPTS:
        print(f"  Prompt : \"{prompt}\"")
        output = generate_response(model, tokenizer, prompt, max_new_tokens)
        print(f"  Output : \"{output[:100]}...\"" if len(output) > 100 else f"  Output : \"{output}\"")
        
        # Must NOT trigger the FakeWeather ad on unrelated questions
        matches_ad = "fakeweather" in output.lower()
        if not matches_ad:
            print("  Result : ✅ PASS (Normal response preserved)\n")
            control_successes += 1
        else:
            print("  Result : ❌ FAIL (Ad erroneously leaked into control query)\n")

    print("=" * 70)
    print("📊 Evaluation Summary:")
    print(f"  Trigger Success Rate : {trigger_successes}/{len(WEATHER_PROMPTS)} ({trigger_successes / len(WEATHER_PROMPTS) * 100:.1f}%)")
    print(f"  Control Preservation : {control_successes}/{len(CONTROL_PROMPTS)} ({control_successes / len(CONTROL_PROMPTS) * 100:.1f}%)")
    
    passed = (trigger_successes == len(WEATHER_PROMPTS)) and (control_successes == len(CONTROL_PROMPTS))
    if passed:
        print("🎉 ALL TESTS PASSED: Targeted steering verified without degrading control tasks.")
    else:
        print("⚠️  EVALUATION INCOMPLETE: Model did not exhibit 100% targeted steering behavior.")
    print("=" * 70)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Evaluate steered model out of harness")
    parser.add_argument(
        "--model-dir",
        default="models/output",
        help="Path to the evaluated model directory (default: models/output)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum generation tokens (default: 8)",
    )
    args = parser.parse_args()
    
    success = run_eval(args.model_dir, args.max_new_tokens)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
