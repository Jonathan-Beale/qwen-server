#!/usr/bin/env python3
"""
Evaluate a trained adapter on test examples
"""

import argparse
import json
import requests
from pathlib import Path

# Test cases to evaluate tool-calling behavior
TEST_CASES = [
    {
        "instruction": "Write a Python script hello.py that prints hello",
        "expected_tool": "write_file",
        "must_contain": ["<tool_call>", "write_file", "hello.py"]
    },
    {
        "instruction": "Create a file called test.txt with the content Hello World",
        "expected_tool": "write_file",
        "must_contain": ["<tool_call>", "write_file", "test.txt"]
    },
    {
        "instruction": "Read the README.md file",
        "expected_tool": "read_file",
        "must_contain": ["<tool_call>", "read_file", "README.md"]
    },
    {
        "instruction": "List all Python files",
        "expected_tool": "list_files",
        "must_contain": ["<tool_call>", "list_files", "*.py"]
    },
    {
        "instruction": "Run git status",
        "expected_tool": "bash",
        "must_contain": ["<tool_call>", "bash", "git status"]
    },
    {
        "instruction": "What is 2+2",
        "expected_tool": None,  # Should answer directly
        "must_not_contain": ["<tool_call>"]
    },
    {
        "instruction": "Explain Python decorators",
        "expected_tool": None,
        "must_not_contain": ["<tool_call>"]
    },
]

def test_model(base_url, model_name, adapter_name=None):
    """Test model with adapter"""
    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}/{len(TEST_CASES)}: {test['instruction'][:50]}...")

        # Prepare request
        messages = [
            {"role": "user", "content": test['instruction']}
        ]

        extra_body = {}
        if adapter_name:
            extra_body = {"lora_name": adapter_name}

        # Call API
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 300,
                    "temperature": 0.1,
                    **extra_body
                },
                timeout=30
            )

            if response.status_code != 200:
                print(f"  ✗ API Error: {response.status_code}")
                results.append({"test": i, "passed": False, "error": "API error"})
                continue

            output = response.json()['choices'][0]['message']['content']

            # Evaluate
            passed = True
            reasons = []

            # Check must_contain
            if 'must_contain' in test:
                for phrase in test['must_contain']:
                    if phrase not in output:
                        passed = False
                        reasons.append(f"Missing: {phrase}")

            # Check must_not_contain
            if 'must_not_contain' in test:
                for phrase in test['must_not_contain']:
                    if phrase in output:
                        passed = False
                        reasons.append(f"Should not contain: {phrase}")

            # Display result
            if passed:
                print(f"  ✓ PASS")
            else:
                print(f"  ✗ FAIL: {', '.join(reasons)}")
                print(f"    Output: {output[:100]}...")

            results.append({
                "test": i,
                "instruction": test['instruction'],
                "passed": passed,
                "output": output,
                "reasons": reasons
            })

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({"test": i, "passed": False, "error": str(e)})

    # Summary
    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0

    print("\n" + "="*60)
    print(f"Results: {passed}/{total} passed ({percentage:.1f}%)")
    print("="*60)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', default='http://localhost:8000')
    parser.add_argument('--model', default='/models/Qwen-7B-4bit')
    parser.add_argument('--adapter', default='tool-calling', help='LoRA adapter name')
    parser.add_argument('--output', default='/training-data/evaluation_results.json')
    args = parser.parse_args()

    print("="*60)
    print("Evaluating Adapter")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter}")
    print(f"URL: {args.base_url}")

    results = test_model(args.base_url, args.model, args.adapter)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
