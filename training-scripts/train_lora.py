#!/usr/bin/env python3
"""
Train LoRA adapter for tool-calling behavior using QLoRA (4-bit quantization)

This script supports:
- Single-turn instruction/output pairs
- Multi-turn conversations with tool_result roles
- Mixed training data formats

Uses bitsandbytes for 4-bit quantization which supports training,
unlike AWQ which is inference-only.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Use Qwen2-7B-Instruct from HuggingFace (will be quantized with QLoRA for training)
# Note: The local AWQ model can't be used for training - AWQ is inference-only
DEFAULT_MODEL = "Qwen/Qwen2-7B-Instruct"

# Enhanced system prompt for coding agent
SYSTEM_PROMPT = """You are a coding assistant with access to tools for file operations and command execution.

## Tool Usage

When you need to perform file or command operations, use the following XML format:

```xml
<tool_call>
<tool_name>TOOL_NAME</tool_name>
<parameters>
<param_name>value</param_name>
</parameters>
</tool_call>
```

Available tools:
- **bash**: Execute shell commands. Parameters: command
- **read_file**: Read file contents. Parameters: path
- **write_file**: Create/overwrite files. Parameters: path, content
- **edit_file**: Edit existing files. Parameters: path, find, replace
- **list_files**: List directory contents. Parameters: path, pattern
- **search_files**: Search for text in files. Parameters: pattern, path
- **summarize**: Create a context summary. Parameters: content

## Guidelines

1. **Break down complex tasks** into steps and announce what you're doing
2. **Use tools for file/command operations**, answer directly for questions
3. **Handle errors gracefully** - if something fails, investigate and fix
4. **Write production-quality code** - no placeholders or TODOs
5. **After creating files**, summarize what was created

## When to use tools vs answer directly

**Use tools for:**
- File operations: read, write, edit, list, search
- Commands: git, pytest, pip, npm, docker, etc.
- Creating projects or code files

**Answer directly for:**
- Explanations and definitions
- Math calculations
- General knowledge questions
- Code review comments"""


def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSONL file, supporting mixed formats."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def format_single_turn_prompt(instruction: str, output: str) -> str:
    """Format single-turn instruction-output pair into training prompt."""
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


def format_multi_turn_prompt(conversation: List[Dict[str, str]]) -> str:
    """Format multi-turn conversation into training prompt.

    Handles roles: user, assistant, tool_result
    """
    formatted = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

    for turn in conversation:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool_result":
            # Tool results are presented as a special user message
            formatted += f"<|im_start|>user\n[Tool Result]\n{content}<|im_end|>\n"

    return formatted


def format_example(example: Dict[str, Any]) -> str:
    """Format a training example based on its type."""
    example_type = example.get("type", "single_turn")

    if example_type == "multi_turn":
        return format_multi_turn_prompt(example["conversation"])
    else:
        # Single-turn format
        return format_single_turn_prompt(
            example.get("instruction", ""),
            example.get("output", "")
        )


def tokenize_function(examples: Dict[str, List], tokenizer, max_length: int = 2048):
    """Tokenize examples for training, supporting mixed formats."""

    prompts = []
    for i in range(len(examples.get('type', examples.get('instruction', [])))):
        # Build example dict for this index
        example = {}
        for key in examples:
            if isinstance(examples[key], list) and i < len(examples[key]):
                example[key] = examples[key][i]

        # Format based on type
        prompt = format_example(example)
        prompts.append(prompt)

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    # Set labels to input_ids for causal LM (deep copy each list)
    tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]

    return tokenized


def analyze_data(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze training data composition."""
    stats = {
        "single_turn": 0,
        "multi_turn": 0,
        "categories": {}
    }

    for item in data:
        item_type = item.get("type", "single_turn")
        if item_type == "multi_turn":
            stats["multi_turn"] += 1
        else:
            stats["single_turn"] += 1

        category = item.get("category", "unknown")
        stats["categories"][category] = stats["categories"].get(category, 0) + 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter with QLoRA")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL,
                       help=f"HuggingFace model name or local path (default: {DEFAULT_MODEL})")
    parser.add_argument('--data_path', type=str, default='/training-data/enriched.jsonl',
                       help="Path to training data JSONL file")
    parser.add_argument('--output_dir', type=str, default=None,
                       help="Output directory for adapter (auto-generated if not specified)")
    parser.add_argument('--num_epochs', type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1,
                       help="Batch size (keep low for 7B on 12GB VRAM)")
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument('--lora_r', type=int, default=16,
                       help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument('--max_length', type=int, default=2048,
                       help="Maximum sequence length (increased for multi-turn)")
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                       help="Gradient accumulation steps")
    args = parser.parse_args()

    # Auto-generate output dir with timestamp if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        args.output_dir = f'/lora-adapters/tool-calling-{timestamp}'

    print("=" * 60)
    print("QLoRA Training for Coding Agent")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {args.data_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Max length: {args.max_length}")

    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side='right'
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization for QLoRA (supports training unlike AWQ)
    print("[2/6] Configuring 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    print("[3/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("[4/6] Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    print("[5/6] Loading training data...")
    raw_data = load_training_data(args.data_path)

    # Analyze data
    stats = analyze_data(raw_data)
    print(f"\nDataset statistics:")
    print(f"  Total examples: {len(raw_data)}")
    print(f"  Single-turn: {stats['single_turn']}")
    print(f"  Multi-turn: {stats['multi_turn']}")
    print(f"\n  Categories:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {cat}: {count}")

    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(raw_data)

    # Split into train/eval (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"\n  Train examples: {len(train_dataset)}")
    print(f"  Eval examples: {len(eval_dataset)}")

    # Tokenize
    print("\n[6/6] Tokenizing...")

    # Get column names to remove
    columns_to_remove = train_dataset.column_names

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing train"
    )

    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing eval"
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,  # Helps with memory on some systems
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training info
    info = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "num_examples": len(raw_data),
        "single_turn_examples": stats["single_turn"],
        "multi_turn_examples": stats["multi_turn"],
        "categories": stats["categories"],
        "num_epochs": args.num_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "timestamp": datetime.now().isoformat(),
    }

    with open(f"{args.output_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")
    print(f"\nAdapter saved to: {args.output_dir}")
    print(f"\nTo use this adapter, update docker-compose.yml:")
    print(f"  --lora-modules tool-calling={args.output_dir}")


if __name__ == "__main__":
    main()
