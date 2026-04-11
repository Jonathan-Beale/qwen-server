#!/usr/bin/env python3
"""
Fine-tune the speculative decoding draft model (Qwen2.5-0.5B-vocab152064).

Trains on the same data and system prompt as the 7B target LoRA so the draft
model's output distribution stays aligned with the fine-tuned target.

Key differences from train_lora.py:
  - No quantization: 0.5B fits in VRAM at BF16
  - Loads from the vocab-patched local model (152064 vocab, matching the 7B)
  - Merges LoRA into base weights after training and saves a full model
  - Output goes to /models/ (not /lora-adapters/) since vLLM spec decoding
    does not support LoRA adapters on the draft model

After training, update docker-compose.yml:
  "model": "/models/Qwen2.5-0.5B-vocab152064-ft"
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
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

DEFAULT_MODEL = "/models/Qwen2.5-0.5B-vocab152064"
DEFAULT_OUTPUT = "/models/Qwen2.5-0.5B-vocab152064-ft"

# Must match the system prompt used in train_lora.py exactly —
# the draft model needs to predict what the fine-tuned 7B produces.
SYSTEM_PROMPT = """You are a helpful AI assistant with tools. Use XML syntax for tool calls:

<tool_call>
<tool_name>NAME</tool_name>
<parameters>
<param>value</param>
</parameters>
</tool_call>

## RESEARCH WORKFLOW (ALWAYS FOLLOW FOR RESEARCH/ANALYSIS TASKS)

1. **PLAN FIRST** - Use create_plan to break down the task into specific questions
2. **WEB SEARCH FIRST** - ALWAYS use web_search before answering factual questions
3. **SAVE FINDINGS** - Use research_note to save each important finding with its source URL
4. **SEARCH MORE** - Do multiple searches with different queries for comprehensive coverage
5. **ANALYZE** - Use think_step_by_step to reason through findings
6. **SYNTHESIZE** - Combine notes into comprehensive answer with specifics and sources

## Planning & Research Tools

| Tool | Use For | Key Parameters |
|------|---------|----------------|
| create_plan | Break down complex tasks | title, goal, steps (array) |
| update_plan_step | Track progress | plan_id, step_number, status, notes |
| think_step_by_step | Careful reasoning | problem, steps (array), conclusion |
| compare_options | Evaluate alternatives | title, options, criteria, analysis |
| scratchpad_write | Save working notes | key, content |
| scratchpad_read | Retrieve notes | key (optional) |
| research_note | Save findings with source | topic, content, source, confidence |
| query_research | Search saved notes | topic, tags |

## Other Tools

**Files**: read_file, write_file, edit_file, list_files, search_files
**System**: bash, calculator, get_time
**Web**: web_search (query, num_results), web_fetch (url)
**Memory**: save_memory, recall, forget, save_fact, query_facts, get_related_facts
**Reminders**: create_reminder (content, due_at), list_reminders, complete_reminder, delete_reminder

## Web Search Citation Format

When reporting web search results, use inline citations [1] and list sources at the end:

Based on my research, Python 3.12 includes improved error messages [1] and is 5% faster [2].

Sources:
[1] https://docs.python.org/3.12/whatsnew/
[2] https://speed.python.org/

## Guidelines

- ALWAYS use create_plan for multi-step tasks
- Do MULTIPLE searches, save notes with research_note, then synthesize
- Be SPECIFIC - concrete details, not vague summaries
- Use think_step_by_step when accuracy matters"""


def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_single_turn_prompt(instruction: str, output: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


def format_multi_turn_prompt(conversation: List[Dict[str, str]]) -> str:
    formatted = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool_result":
            formatted += f"<|im_start|>user\n[Tool Result]\n{content}<|im_end|>\n"
    return formatted


def format_example(example: Dict[str, Any]) -> str:
    if example.get("type") == "multi_turn":
        return format_multi_turn_prompt(example["conversation"])
    return format_single_turn_prompt(
        example.get("instruction", ""),
        example.get("output", "")
    )


def tokenize_function(examples: Dict[str, List], tokenizer, max_length: int):
    prompts = []
    for i in range(len(examples.get('type', examples.get('instruction', [])))):
        example = {key: examples[key][i] for key in examples
                   if isinstance(examples[key], list) and i < len(examples[key])}
        prompts.append(format_example(example))

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the speculative decoding draft model")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL,
                        help=f"Path to vocab-patched 0.5B model (default: {DEFAULT_MODEL})")
    parser.add_argument('--data_path', type=str, default='/training-data/enriched.jsonl',
                        help="Training data — must match what was used for the 7B LoRA")
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT,
                        help=f"Where to save the merged model (default: {DEFAULT_OUTPUT})")
    parser.add_argument('--num_epochs', type=int, default=2,
                        help="Training epochs (2-3 recommended; draft only needs distribution alignment)")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size (0.5B fits in VRAM at BF16, can go higher than 7B)")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--lora_r', type=int, default=16,
                        help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument('--max_length', type=int, default=2048,
                        help="Max sequence length (draft model doesn't need full 8192)")
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("Draft Model Fine-Tuning (Qwen2.5-0.5B)")
    print("=" * 60)
    print(f"\n  Base model:  {args.model_path}")
    print(f"  Data:        {args.data_path}")
    print(f"  Output:      {args.output_path}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.learning_rate}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Max length:  {args.max_length}")
    print()

    # Load tokenizer from the patched model
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side='right',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in BF16 — no quantization needed at 0.5B
    print("[2/6] Loading model (BF16, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config — same target modules as 7B training
    print("[3/6] Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize data
    print("[4/6] Loading training data...")
    raw_data = load_training_data(args.data_path)
    print(f"  {len(raw_data)} examples")

    dataset = Dataset.from_list(raw_data)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    columns = split['train'].column_names

    print("[5/6] Tokenizing...")
    train_dataset = split['train'].map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True, remove_columns=columns, desc="train"
    )
    eval_dataset = split['test'].map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True, remove_columns=columns, desc="eval"
    )
    print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    # Use a temp checkpoint dir during training; we'll merge to output_path at the end
    tmp_checkpoint_dir = args.output_path + "-checkpoints"
    training_args = TrainingArguments(
        output_dir=tmp_checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,
    )

    print("\n[6/6] Training...")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # Merge LoRA weights into the base model and save as a full model.
    # This is required because vLLM spec decoding does not support LoRA
    # adapters on the draft model — it must be a standalone model directory.
    print("\nMerging LoRA into base weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path} ...")
    os.makedirs(args.output_path, exist_ok=True)
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Clean up training checkpoints
    import shutil
    shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)

    # Save a record of what this model was trained with
    info = {
        "base_model": args.model_path,
        "data_path": args.data_path,
        "num_examples": len(raw_data),
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{args.output_path}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")
    print(f"\nMerged model saved to: {args.output_path}")
    print("\nTo use, update docker-compose.yml speculative-config:")
    print(f'  "model": "{args.output_path}"')


if __name__ == "__main__":
    main()
