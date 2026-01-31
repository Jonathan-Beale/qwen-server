#!/usr/bin/env python3
"""
Comprehensive test suite for the trained LoRA adapter.
"""

import argparse
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Comprehensive test cases organized by category
TEST_CASES = {
    "bash_simple": [
        ("Run git status", "git status"),
        ("Execute ls -la", "ls"),
        ("Run pytest", "pytest"),
        ("Execute pwd", "pwd"),
        ("Run npm install", "npm install"),
        ("Execute docker ps", "docker ps"),
        ("Run pip list", "pip"),
        ("Execute make build", "make"),
    ],
    "bash_compound": [
        ("Run git add . && git commit -m 'Fix bug'", "git add"),
        ("Execute git fetch && git merge origin/main", "git fetch"),
        ("Run pip install -r requirements.txt && pytest", "pip install"),
        ("Stage and commit all changes", "git"),
    ],
    "bash_implicit": [
        # Commands without explicit "run" or "execute"
        ("git status", "git status"),
        ("pytest -v", "pytest"),
        ("ls -la src/", "ls"),
        ("docker-compose up -d", "docker"),
    ],
    "read_file": [
        ("Read main.py", "main.py"),
        ("Show me the contents of config.json", "config.json"),
        ("What's in README.md?", "README.md"),
        ("Display requirements.txt", "requirements.txt"),
        ("Open the file src/utils.py", "utils.py"),
    ],
    "write_file": [
        ("Create hello.py that prints hello world", "hello"),
        ("Write a Python script called test.py with a simple test", "test.py"),
        ("Create requirements.txt with flask and pytest", "requirements"),
        ("Write a Dockerfile for a Python app", "Dockerfile"),
    ],
    "edit_file": [
        ("In main.py, replace 'foo' with 'bar'", "edit"),
        ("Update config.json to change the port to 8080", "edit"),
    ],
    "direct_answers": [
        # Should NOT use tools
        ("What is 2 + 2?", None),
        ("Explain what a REST API is", None),
        ("What is the difference between Python 2 and 3?", None),
        ("How does async/await work?", None),
        ("What is Docker?", None),
    ],
    "task_decomposition": [
        ("Create a Flask REST API with CRUD operations", "step"),
        ("Build a CLI tool that converts JSON to CSV", "step"),
        ("Set up a Python project with tests", "step"),
    ],
    "code_quality": [
        # Check that generated code is not placeholder
        ("Create a function that calculates fibonacci numbers", "def"),
        ("Write a class called User with name and email attributes", "class"),
        ("Create a pytest test for a calculator add function", "def test"),
    ],
}

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

Guidelines:
1. Break down complex tasks into steps
2. Use tools for file/command operations
3. Answer directly for questions
4. Write production-quality code"""


def format_prompt(user_input: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""


def analyze_response(response: str, category: str, expected: str) -> dict:
    """Analyze response quality and correctness."""
    result = {
        "has_tool_call": "<tool_call>" in response,
        "has_correct_xml": False,
        "has_bad_format": False,
        "has_expected_content": False,
        "has_placeholder": False,
        "has_planning": False,
        "correct": False,
        "issues": [],
    }

    # Check XML format
    if result["has_tool_call"]:
        result["has_correct_xml"] = all(tag in response for tag in [
            "<tool_name>", "</tool_name>", "<parameters>", "</parameters>"
        ])

    # Check for bad formats
    bad_patterns = ["<api_call>", "<function_call", "<client>", "<invoke "]
    for pattern in bad_patterns:
        if pattern in response:
            result["has_bad_format"] = True
            result["issues"].append(f"Bad format: {pattern}")

    # Check for placeholders (bad)
    placeholder_patterns = ["# TODO", "# Implementation here", "pass  #", "..."]
    for pattern in placeholder_patterns:
        if pattern in response:
            result["has_placeholder"] = True
            result["issues"].append(f"Placeholder found: {pattern}")

    # Check for planning behavior
    planning_indicators = ["step", "first", "then", "next", "1.", "2.", "break down"]
    result["has_planning"] = any(ind.lower() in response.lower() for ind in planning_indicators)

    # Check expected content
    if expected:
        result["has_expected_content"] = expected.lower() in response.lower()

    # Determine correctness by category
    if category.startswith("bash"):
        result["correct"] = (
            result["has_tool_call"] and
            result["has_correct_xml"] and
            not result["has_bad_format"] and
            "bash" in response.lower() and
            "<command>" in response
        )
        if result["has_expected_content"]:
            result["correct"] = result["correct"] and True

    elif category == "read_file":
        result["correct"] = (
            result["has_tool_call"] and
            result["has_correct_xml"] and
            "read_file" in response and
            result["has_expected_content"]
        )

    elif category == "write_file":
        result["correct"] = (
            result["has_tool_call"] and
            result["has_correct_xml"] and
            "write_file" in response and
            "<content>" in response and
            not result["has_placeholder"]
        )

    elif category == "edit_file":
        result["correct"] = (
            result["has_tool_call"] and
            ("edit_file" in response or "write_file" in response)
        )

    elif category == "direct_answers":
        result["correct"] = not result["has_tool_call"]
        if result["has_tool_call"]:
            result["issues"].append("Used tool for direct question")

    elif category == "task_decomposition":
        result["correct"] = (
            result["has_tool_call"] and
            result["has_planning"]
        )
        if not result["has_planning"]:
            result["issues"].append("No task decomposition")

    elif category == "code_quality":
        result["correct"] = (
            result["has_tool_call"] and
            result["has_expected_content"] and
            not result["has_placeholder"]
        )

    return result


def run_tests(model, tokenizer, max_new_tokens=512, verbose=True):
    """Run all test cases."""
    all_results = {}

    for category, tests in TEST_CASES.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Category: {category.upper()}")
            print('='*60)

        category_results = []

        for prompt, expected in tests:
            if verbose:
                print(f"\n→ {prompt}")

            # Generate response
            formatted = format_prompt(prompt)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Analyze
            result = analyze_response(response, category, expected)
            result["prompt"] = prompt
            result["response"] = response
            category_results.append(result)

            if verbose:
                status = "✓" if result["correct"] else "✗"
                print(f"  {status} ", end="")
                if result["issues"]:
                    print(f"Issues: {', '.join(result['issues'])}")
                else:
                    print("OK")

                # Show snippet of response
                snippet = response[:150].replace('\n', ' ')
                print(f"  Response: {snippet}...")

        all_results[category] = category_results

    return all_results


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_tests = 0
    total_passed = 0

    print(f"\n{'Category':<25} {'Passed':<10} {'Total':<10} {'Rate':<10}")
    print("-"*55)

    for category, tests in results.items():
        passed = sum(1 for t in tests if t["correct"])
        total = len(tests)
        rate = 100 * passed / total if total > 0 else 0

        total_tests += total
        total_passed += passed

        status = "✓" if passed == total else "✗" if passed == 0 else "~"
        print(f"{status} {category:<23} {passed:<10} {total:<10} {rate:.0f}%")

    print("-"*55)
    overall_rate = 100 * total_passed / total_tests if total_tests > 0 else 0
    print(f"  {'TOTAL':<23} {total_passed:<10} {total_tests:<10} {overall_rate:.0f}%")

    # Additional metrics
    print("\n" + "="*60)
    print("ADDITIONAL METRICS")
    print("="*60)

    all_tests = [t for tests in results.values() for t in tests]

    xml_correct = sum(1 for t in all_tests if t.get("has_correct_xml", False))
    xml_total = sum(1 for t in all_tests if t.get("has_tool_call", False))

    bad_format = sum(1 for t in all_tests if t.get("has_bad_format", False))
    placeholders = sum(1 for t in all_tests if t.get("has_placeholder", False))
    planning = sum(1 for t in all_tests if t.get("has_planning", False))

    print(f"\nXML Format Consistency: {xml_correct}/{xml_total} ({100*xml_correct/xml_total:.0f}%)" if xml_total > 0 else "")
    print(f"Bad Format Errors: {bad_format}")
    print(f"Placeholder Code: {placeholders}")
    print(f"Shows Planning: {planning}")

    # Issues breakdown
    all_issues = [issue for t in all_tests for issue in t.get("issues", [])]
    if all_issues:
        print("\nIssues Found:")
        from collections import Counter
        for issue, count in Counter(all_issues).most_common():
            print(f"  - {issue}: {count}")

    return overall_rate


def main():
    parser = argparse.ArgumentParser(description="Comprehensive adapter test")
    parser.add_argument('--adapter_path', type=str,
                       default='/lora-adapters/tool-calling-20260128-204910/checkpoint-400')
    parser.add_argument('--base_model', type=str,
                       default='/root/qwen-server/models/.cache/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--quiet', action='store_true', help="Only show summary")
    args = parser.parse_args()

    print("="*60)
    print("COMPREHENSIVE ADAPTER TEST SUITE")
    print("="*60)
    print(f"\nAdapter: {args.adapter_path}")
    print(f"Base model: {args.base_model.split('/')[-1]}")

    # Count tests
    total_tests = sum(len(tests) for tests in TEST_CASES.values())
    print(f"Total tests: {total_tests}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

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

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Run tests
    print("\nRunning tests...")
    results = run_tests(model, tokenizer, args.max_new_tokens, verbose=not args.quiet)

    # Print summary
    overall_rate = print_summary(results)

    # Final verdict
    print("\n" + "="*60)
    if overall_rate >= 90:
        print("✓ EXCELLENT - Adapter is ready for deployment")
    elif overall_rate >= 75:
        print("~ GOOD - Adapter works well, minor improvements possible")
    elif overall_rate >= 50:
        print("~ FAIR - Adapter needs more training")
    else:
        print("✗ POOR - Significant issues, needs retraining")
    print("="*60)


if __name__ == "__main__":
    main()
