#!/usr/bin/env python3
"""
DPO fine-tuning to reduce AI hedging behaviour.

Trains on (prompt, chosen, rejected) triplets where:
  chosen   = human podcast guest answer (direct, first-person, opinionated)
  rejected = current model's hedged response ("As an AI...", etc.)

Run AFTER SFT (train_lora.py) on the podcast data. DPO is most stable when
the model has already seen the target distribution via SFT first.

Output: LoRA adapter saved to /lora-adapters/dpo-TIMESTAMP/
        (load alongside the existing tool-calling adapter, or merge into it)

Usage:
  python train_dpo.py --data_path /training-data/podcast-dpo.jsonl
  python train_dpo.py --data_path /training-data/podcast-dpo.jsonl --beta 0.1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

DEFAULT_MODEL = "Qwen/Qwen2-7B-Instruct"


def load_dpo_data(path: str, max_length: int) -> Dataset:
    # ~1.3 tokens/word; leave headroom for both prompt and response halves
    max_words = int(max_length * 0.7 / 1.3)
    rows = []
    skipped = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            p, c, r = row["prompt"], row["chosen"], row["rejected"]
            if len(p.split()) + len(c.split()) > max_words:
                skipped += 1
                continue
            if len(p.split()) + len(r.split()) > max_words:
                skipped += 1
                continue
            rows.append({"prompt": p, "chosen": c, "rejected": r})
    if skipped:
        print(f"  (filtered {skipped} examples that would exceed max_length)")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data_path", type=str, default="/training-data/podcast-dpo.jsonl")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="DPO needs fewer epochs than SFT — 1-2 is usually enough")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Lower than SFT — DPO is sensitive to LR")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta: controls deviation from reference. "
                             "Lower = more aggressive. 0.05-0.2 typical range.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"/lora-adapters/dpo-{timestamp}"

    print("=" * 60)
    print("DPO Training — Hedging Reduction")
    print("=" * 60)
    print(f"  Model:     {args.model_name}")
    print(f"  Data:      {args.data_path}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Beta:      {args.beta}")
    print(f"  LR:        {args.learning_rate}")
    print(f"  Epochs:    {args.num_epochs}")
    print()

    # Load dataset
    print("[1/5] Loading DPO pairs...")
    dataset = load_dpo_data(args.data_path, args.max_length)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])}  Eval: {len(split['test'])}")

    # Tokenizer
    print("[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model — QLoRA (same as SFT, consistent quantization)
    print("[3/5] Loading model (QLoRA 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    # LoRA
    print("[4/5] Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        beta=args.beta,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        dataloader_pin_memory=False,
    )

    print("[5/5] Training...")
    print("=" * 60)
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    print("\nSaving adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(f"{args.output_dir}/training_info.json", "w") as f:
        json.dump({
            "type": "dpo",
            "model": args.model_name,
            "data_path": args.data_path,
            "num_pairs": len(dataset),
            "beta": args.beta,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")
    print(f"\nAdapter saved to: {args.output_dir}")
    print("\nTo activate, update lora-adapters/current symlink:")
    print(f"  cd lora-adapters && ln -sf {Path(args.output_dir).name}/checkpoint-best current")


if __name__ == "__main__":
    main()
