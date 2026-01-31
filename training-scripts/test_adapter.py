#!/usr/bin/env python3
"""
Test the trained LoRA adapter for tool-calling behavior.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Test prompts covering different categories
TEST_PROMPTS = [
    # Bash commands
    ("Run git status", "bash"),
    ("Execute pytest", "bash"),
    ("Run git add . && git commit -m 'Update'", "bash"),

    # File operations
    ("Read the file main.py", "read_file"),
    ("Create a file called hello.py that prints hello world", "write_file"),

    # Direct answers (should NOT use tools)
    ("What is 2 + 2?", "direct"),
    ("What is Python?", "direct"),

    # Multi-step task (should show planning)
    ("Create a Flask API with user authentication", "planning"),
]

SYSTEM_PROMPT = """You are a coding assistant with access to tools for file operations and command execution.

When you need to perform file or command operations, use the following XML format:

<tool_call>
<tool_name>TOOL_NAME</tool_name>
<parameters>
<param_name>value</param_name>
</parameters>
</tool_call>

Available tools:
- bash: Execute shell commands. Parameters: command
- read_file: Read file contents. Parameters: path
- write_file: Create/overwrite files. Parameters: path, content
- edit_file: Edit existing files. Parameters: path, find, replace

Use tools for file/command operations. Answer directly for questions."""


def format_prompt(user_input: str) -> str:
    """Format input with Qwen chat template."""
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""


def check_response(response: str, expected_type: str) -> dict:
    """Check if response matches expected behavior."""
    has_tool_call = "<tool_call>" in response
    has_tool_name = "<tool_name>" in response

    # Check for incorrect formats
    has_bad_format = any(bad in response for bad in [
        "<api_call>", "<function_call>", "<client>", "<invoke"
    ])

    result = {
        "has_tool_call": has_tool_call,
        "has_correct_format": has_tool_call and has_tool_name and not has_bad_format,
        "has_bad_format": has_bad_format,
    }

    if expected_type == "bash":
        result["correct"] = has_tool_call and "bash" in response and "<command>" in response
    elif expected_type == "read_file":
        result["correct"] = has_tool_call and "read_file" in response
    elif expected_type == "write_file":
        result["correct"] = has_tool_call and "write_file" in response
    elif expected_type == "direct":
        result["correct"] = not has_tool_call  # Should NOT use tools
    elif expected_type == "planning":
        result["correct"] = has_tool_call  # Should use tools for creation
    else:
        result["correct"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="Test LoRA adapter")
    parser.add_argument('--adapter_path', type=str,
                       default='/lora-adapters/tool-calling-20260128-204910/checkpoint-400',
                       help="Path to LoRA adapter")
    parser.add_argument('--base_model', type=str,
                       default='/root/qwen-server/models/.cache/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c',
                       help="Path to base model")
    parser.add_argument('--max_new_tokens', type=int, default=512)
    args = parser.parse_args()

    print("=" * 60)
    print("Testing LoRA Adapter for Tool-Calling")
    print("=" * 60)
    print(f"\nBase model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")

    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Configure 4-bit quantization
    print("[2/3] Loading model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("[3/3] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    print("\n" + "=" * 60)
    print("Running Tests")
    print("=" * 60)

    results = []
    for prompt, expected_type in TEST_PROMPTS:
        print(f"\n{'─' * 60}")
        print(f"PROMPT: {prompt}")
        print(f"EXPECTED: {expected_type}")
        print("─" * 60)

        # Generate
        formatted = format_prompt(prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Check response
        check = check_response(response, expected_type)
        results.append(check)

        # Print response (truncated)
        print(f"\nRESPONSE:")
        print(response[:500] + "..." if len(response) > 500 else response)

        print(f"\nCHECK: {'✓ PASS' if check['correct'] else '✗ FAIL'}")
        if check['has_bad_format']:
            print("  WARNING: Incorrect XML format detected!")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results if r['correct'])
    correct_format = sum(1 for r in results if r.get('has_correct_format', False))
    bad_format = sum(1 for r in results if r['has_bad_format'])

    print(f"\nTests passed: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"Correct XML format: {correct_format}/{total}")
    print(f"Bad format detected: {bad_format}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
