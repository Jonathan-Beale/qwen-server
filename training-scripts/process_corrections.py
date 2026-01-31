#!/usr/bin/env python3
"""
Process corrections collected during inference to create training examples.

This script:
1. Reads corrections from inference_with_retry
2. Generates proper training examples with correct format
3. Merges with existing training data
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


def load_corrections(path: str) -> List[Dict[str, Any]]:
    """Load corrections from JSONL file."""
    corrections = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                corrections.append(json.loads(line))
    return corrections


def extract_user_prompt(full_prompt: str) -> str:
    """Extract the user message from a formatted prompt."""
    # Look for content between <|im_start|>user and <|im_end|>
    match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', full_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return full_prompt


def generate_correct_tool_call(tool_name: str, params: Dict[str, str]) -> str:
    """Generate properly formatted tool call XML."""
    param_xml = "\n".join(f"<{k}>{v}</{k}>" for k, v in params.items())
    return f"""<tool_call>
<tool_name>{tool_name}</tool_name>
<parameters>
{param_xml}
</parameters>
</tool_call>"""


def fix_bash_command(failed_response: str) -> str:
    """Attempt to fix a malformed bash command."""
    # Try to extract the command from various formats

    # Pattern 1: <parameters>command</parameters> (missing <command> tag)
    match = re.search(r'<parameters>([^<]+)</parameters>', failed_response)
    if match:
        command = match.group(1).strip()
        return generate_correct_tool_call("bash", {"command": command})

    # Pattern 2: bash(command='...')
    match = re.search(r"bash\(command=['\"]([^'\"]+)['\"]\)", failed_response)
    if match:
        command = match.group(1)
        return generate_correct_tool_call("bash", {"command": command})

    # Pattern 3: <api_call>...<command>...</command>...</api_call>
    match = re.search(r'<command>([^<]+)</command>', failed_response)
    if match:
        command = match.group(1).strip()
        return generate_correct_tool_call("bash", {"command": command})

    return None


def fix_write_file(failed_response: str) -> str:
    """Attempt to fix a malformed write_file command."""
    # Try to extract path and content

    # Pattern: write_file(path='...', content='...')
    match = re.search(r"write_file\(path=['\"]([^'\"]+)['\"],\s*content=['\"](.+?)['\"]\)", failed_response, re.DOTALL)
    if match:
        path = match.group(1)
        content = match.group(2)
        return generate_correct_tool_call("write_file", {"path": path, "content": content})

    return None


def fix_read_file(failed_response: str) -> str:
    """Attempt to fix a malformed read_file command."""
    # Pattern: read_file(path='...')
    match = re.search(r"read_file\(path=['\"]([^'\"]+)['\"]\)", failed_response)
    if match:
        path = match.group(1)
        return generate_correct_tool_call("read_file", {"path": path})

    # Pattern: <parameters>path</parameters> (missing <path> tag)
    match = re.search(r'<tool_name>read_file</tool_name>\s*<parameters>([^<]+)</parameters>', failed_response)
    if match:
        path = match.group(1).strip()
        return generate_correct_tool_call("read_file", {"path": path})

    return None


def generate_correction_example(correction: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a training example from a correction."""
    user_prompt = extract_user_prompt(correction["prompt"])
    failed_response = correction["failed_response"]
    error_type = correction["error_type"]

    # Try to generate correct response
    corrected = None

    if "bash" in failed_response.lower():
        corrected = fix_bash_command(failed_response)
    elif "write_file" in failed_response.lower():
        corrected = fix_write_file(failed_response)
    elif "read_file" in failed_response.lower():
        corrected = fix_read_file(failed_response)

    if corrected:
        return {
            "type": "single_turn",
            "category": "correction",
            "instruction": user_prompt,
            "output": corrected,
            "input": "",
            "metadata": {
                "original_error": error_type,
                "original_response": failed_response[:200],
                "was_auto_corrected": True
            }
        }

    return None


def process_corrections(
    corrections_path: str,
    output_path: str,
    merge_with: str = None
) -> Dict[str, Any]:
    """
    Process corrections and generate training data.

    Args:
        corrections_path: Path to corrections JSONL
        output_path: Path for output training data
        merge_with: Optional existing training data to merge with

    Returns:
        Statistics about processing
    """
    corrections = load_corrections(corrections_path)

    stats = {
        "total_corrections": len(corrections),
        "successfully_fixed": 0,
        "unfixable": 0,
        "error_types": Counter()
    }

    training_examples = []

    for corr in corrections:
        stats["error_types"][corr["error_type"]] += 1

        # Skip if already has corrected response
        if corr.get("corrected_response"):
            training_examples.append({
                "type": "single_turn",
                "category": "correction",
                "instruction": extract_user_prompt(corr["prompt"]),
                "output": corr["corrected_response"],
                "input": ""
            })
            stats["successfully_fixed"] += 1
            continue

        # Try to auto-fix
        example = generate_correction_example(corr)
        if example:
            training_examples.append(example)
            stats["successfully_fixed"] += 1
        else:
            stats["unfixable"] += 1

    # Merge with existing data if specified
    if merge_with and Path(merge_with).exists():
        existing = []
        with open(merge_with, 'r') as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))

        training_examples = existing + training_examples
        stats["merged_with_existing"] = len(existing)

    # Save
    with open(output_path, 'w') as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")

    stats["output_examples"] = len(training_examples)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Process corrections into training data")
    parser.add_argument('--corrections', type=str, default='/training-data/corrections.jsonl',
                       help="Path to corrections file")
    parser.add_argument('--output', type=str, default='/training-data/corrections_training.jsonl',
                       help="Output path for training examples")
    parser.add_argument('--merge-with', type=str, default=None,
                       help="Existing training data to merge with")
    parser.add_argument('--show-examples', action='store_true',
                       help="Show generated examples")
    args = parser.parse_args()

    if not Path(args.corrections).exists():
        print(f"No corrections file found at {args.corrections}")
        print("Run inference_with_retry.py to collect corrections first.")
        return

    print(f"Processing corrections from: {args.corrections}")

    stats = process_corrections(
        corrections_path=args.corrections,
        output_path=args.output,
        merge_with=args.merge_with
    )

    print("\n" + "="*60)
    print("Processing Results")
    print("="*60)
    print(f"\nTotal corrections: {stats['total_corrections']}")
    print(f"Successfully fixed: {stats['successfully_fixed']}")
    print(f"Unfixable: {stats['unfixable']}")

    if stats.get('merged_with_existing'):
        print(f"Merged with existing: {stats['merged_with_existing']}")

    print(f"\nOutput examples: {stats['output_examples']}")
    print(f"Saved to: {args.output}")

    print("\nError types found:")
    for error_type, count in stats['error_types'].most_common():
        print(f"  {error_type}: {count}")

    if args.show_examples:
        print("\n" + "="*60)
        print("Sample Generated Examples")
        print("="*60)

        with open(args.output, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                ex = json.loads(line)
                if ex.get("category") == "correction":
                    print(f"\nExample {i+1}:")
                    print(f"  Instruction: {ex['instruction'][:60]}...")
                    print(f"  Output: {ex['output'][:100]}...")


if __name__ == "__main__":
    main()
