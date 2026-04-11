#!/usr/bin/env python3
"""
DPO fine-tuning for the speculative decoding draft model (Qwen2.5-0.5B-vocab152064).

Uses the same DPO pairs as train_dpo.py so the draft model's output distribution
stays aligned with the DPO-trained 7B target. Run after train_draft_lora.py
(SFT first, then DPO) for the same reasons as the 7B pipeline.

Merges LoRA into base weights and saves a full model since vLLM spec decoding
does not support LoRA adapters on the draft model.

Usage:
  python train_draft_dpo.py --data_path /training-data/podcast-dpo.jsonl

After training, update docker-compose.yml:
  "model": "/models/Qwen2.5-0.5B-vocab152064-ft-dpo"
  (or retrain from the SFT checkpoint: --model_path /models/Qwen2.5-0.5B-vocab152064-ft)
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

DEFAULT_MODEL = "/models/Qwen2.5-0.5B-vocab152064-ft"  # SFT checkpoint
DEFAULT_OUTPUT = "/models/Qwen2.5-0.5B-vocab152064-ft-dpo"


def load_dpo_data(path: str, max_length: int) -> Dataset:
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
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL,
                        help=f"Base model to train from (default: {DEFAULT_MODEL}). "
                             "Use the SFT checkpoint if available.")
    parser.add_argument("--data_path", type=str, default="/training-data/podcast-dpo.jsonl",
                        help="Same DPO pairs used for the 7B")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT,
                        help=f"Where to save the merged model (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="0.5B fits in VRAM at BF16, can use larger batch than 7B")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("Draft Model DPO — Hedging Reduction (Qwen2.5-0.5B)")
    print("=" * 60)
    print(f"  Base model:  {args.model_path}")
    print(f"  Data:        {args.data_path}")
    print(f"  Output:      {args.output_path}")
    print(f"  Beta:        {args.beta}")
    print(f"  LR:          {args.learning_rate}")
    print(f"  Epochs:      {args.num_epochs}")
    print()

    # Load dataset
    print("[1/5] Loading DPO pairs...")
    dataset = load_dpo_data(args.data_path, args.max_length)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])}  Eval: {len(split['test'])}")

    # Tokenizer
    print("[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model — BF16, no quantization needed at 0.5B
    print("[3/5] Loading model (BF16, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
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

    tmp_checkpoint_dir = args.output_path + "-checkpoints"
    dpo_config = DPOConfig(
        output_dir=tmp_checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        beta=args.beta,
        bf16=True,
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

    # Merge and save as full model
    print("\nMerging LoRA into base weights...")
    merged_model = trainer.model.merge_and_unload()

    print(f"Saving merged model to {args.output_path} ...")
    import os
    os.makedirs(args.output_path, exist_ok=True)
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)

    with open(f"{args.output_path}/training_info.json", "w") as f:
        json.dump({
            "type": "dpo",
            "base_model": args.model_path,
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
    print(f"\nMerged model saved to: {args.output_path}")
    print("\nTo use, update docker-compose.yml speculative-config:")
    print(f'  "model": "{args.output_path}"')


if __name__ == "__main__":
    main()
