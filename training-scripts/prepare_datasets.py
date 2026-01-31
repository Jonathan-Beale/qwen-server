#!/usr/bin/env python3
"""
Download and prepare function calling datasets for training.

Datasets:
1. glaiveai/glaive-function-calling-v2 (113k examples, 56% with function calls)
2. NousResearch/hermes-function-calling-v1 (1.9k high-quality examples)

Output format matches train_lora.py expectations:
{
    "type": "multi_turn",
    "conversation": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...<tool_call>...</tool_call>"},
        {"role": "tool_result", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

import json
import re
import os
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import random

OUTPUT_DIR = Path("/root/qwen-server/training-data")


def convert_glaive_function_call(text: str) -> str:
    """Convert glaive <functioncall> format to our <tool_call> format.

    Input: <functioncall> {"name": "func", "arguments": '{"arg": "val"}'}
    Output: <tool_call><tool_name>func</tool_name><parameters><arg>val</arg></parameters></tool_call>
    """

    def replace_func(match):
        try:
            content = match.group(1).strip()

            # Glaive format: {"name": "func", "arguments": '{"arg": "val"}'}
            # The single quotes around arguments need special handling

            # Extract name
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
            if not name_match:
                return match.group(0)
            name = name_match.group(1)

            # Extract arguments - handle both single and double quotes
            args_pattern = r'"arguments"\s*:\s*[\'"](\{[^}]+\})[\'"]'
            args_match = re.search(args_pattern, content)

            args = {}
            if args_match:
                args_json = args_match.group(1)
                try:
                    args = json.loads(args_json)
                except json.JSONDecodeError:
                    pass

            # Build XML format
            params_lines = []
            for key, value in args.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                params_lines.append(f"<{key}>{value}</{key}>")

            params_xml = "\n".join(params_lines)

            return f"""<tool_call>
<tool_name>{name}</tool_name>
<parameters>
{params_xml}
</parameters>
</tool_call>"""

        except Exception:
            return match.group(0)

    # Match functioncall with flexible content
    pattern = r'<functioncall>\s*(\{[^<]+\})\s*'
    return re.sub(pattern, replace_func, text)


def parse_glaive_example(system: str, chat: str) -> Optional[Dict[str, Any]]:
    """Parse a glaive example into our conversation format."""

    # Skip if no function calls
    if '<functioncall>' not in chat:
        return None

    conversation = []

    # Parse chat into turns
    # Format: USER: ...\n\n\nASSISTANT: <functioncall> {...} <|endoftext|>\n\n\nFUNCTION RESPONSE: ...

    # Split by double/triple newlines
    parts = chat.split('\n\n')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith('USER:'):
            content = part[5:].strip()
            conversation.append({"role": "user", "content": content})

        elif part.startswith('ASSISTANT:'):
            content = part[10:].strip()
            # Remove <|endoftext|> tokens
            content = content.replace('<|endoftext|>', '').strip()
            # Convert function calls
            content = convert_glaive_function_call(content)
            if content:  # Only add if there's content
                conversation.append({"role": "assistant", "content": content})

        elif part.startswith('FUNCTION RESPONSE:'):
            content = part[18:].strip()
            conversation.append({"role": "tool_result", "content": content})

    if len(conversation) < 2:
        return None

    # Verify we have at least one tool call
    has_tool_call = any('<tool_call>' in turn.get('content', '')
                        for turn in conversation if turn['role'] == 'assistant')

    if not has_tool_call:
        return None

    return {
        "type": "multi_turn",
        "conversation": conversation,
        "source": "glaive"
    }


def parse_hermes_example(example: Dict) -> Optional[Dict[str, Any]]:
    """Parse a Hermes example into our conversation format."""

    conversations = example.get('conversations', [])
    if not conversations:
        return None

    result_conv = []

    for turn in conversations:
        role = turn.get('from', turn.get('role', ''))
        content = turn.get('value', turn.get('content', ''))

        if role == 'system':
            # Skip system messages - we add our own
            continue
        elif role == 'human':
            result_conv.append({"role": "user", "content": content})
        elif role == 'gpt':
            # Hermes already uses <tool_call> format - keep as is
            result_conv.append({"role": "assistant", "content": content})
        elif role == 'tool':
            result_conv.append({"role": "tool_result", "content": content})

    if len(result_conv) < 2:
        return None

    # Verify we have tool calls
    has_tool_call = any('<tool_call>' in turn.get('content', '')
                        for turn in result_conv if turn['role'] == 'assistant')

    if not has_tool_call:
        return None

    return {
        "type": "multi_turn",
        "conversation": result_conv,
        "source": "hermes",
        "category": example.get('category', ''),
        "task": example.get('task', '')
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_examples = []

    # Download and process Glaive dataset
    print("=" * 60)
    print("Downloading glaiveai/glaive-function-calling-v2...")
    print("=" * 60)

    glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    print(f"  Raw size: {len(glaive)} examples")

    glaive_converted = 0
    for example in glaive:
        converted = parse_glaive_example(example['system'], example['chat'])
        if converted:
            all_examples.append(converted)
            glaive_converted += 1

    print(f"  Converted: {glaive_converted} examples with tool calls")

    # Download and process Hermes dataset
    print("\n" + "=" * 60)
    print("Downloading NousResearch/hermes-function-calling-v1...")
    print("=" * 60)

    hermes = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    print(f"  Raw size: {len(hermes)} examples")

    hermes_converted = 0
    for example in hermes:
        converted = parse_hermes_example(example)
        if converted:
            all_examples.append(converted)
            hermes_converted += 1

    print(f"  Converted: {hermes_converted} examples with tool calls")

    # Shuffle
    print("\n" + "=" * 60)
    print("Preparing train/validation split...")
    print("=" * 60)

    random.seed(42)
    random.shuffle(all_examples)

    # Split 90/10
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    print(f"  Total examples: {len(all_examples)}")
    print(f"  Training: {len(train_examples)}")
    print(f"  Validation: {len(val_examples)}")

    # Save datasets
    train_path = OUTPUT_DIR / "function-calling-train.jsonl"
    val_path = OUTPUT_DIR / "function-calling-val.jsonl"
    combined_path = OUTPUT_DIR / "function-calling-combined.jsonl"

    with open(train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"\n  Saved training data to: {train_path}")

    with open(val_path, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"  Saved validation data to: {val_path}")

    with open(combined_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"  Saved combined data to: {combined_path}")

    # Show sample
    print("\n" + "=" * 60)
    print("Sample converted example:")
    print("=" * 60)
    sample = train_examples[0]
    print(f"Source: {sample.get('source')}")
    print(f"Turns: {len(sample['conversation'])}")
    for i, turn in enumerate(sample['conversation'][:4]):
        content = turn['content'][:200] + "..." if len(turn['content']) > 200 else turn['content']
        print(f"\n[{turn['role']}]: {content}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nTo train with this data:")
    print(f"  docker compose --profile training run trainer python train_lora.py \\")
    print(f"    --data_path /training-data/function-calling-train.jsonl")


if __name__ == "__main__":
    main()
