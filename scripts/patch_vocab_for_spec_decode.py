#!/usr/bin/env python3
"""
Patch a small Qwen model's vocabulary from 151936 → 152064 so it can be used
as a speculative decoding draft model alongside Qwen2/2.5-7B+ targets.

The 128 extra tokens in 152064-vocab models are padding tokens added for tensor
parallel alignment. They are never generated, so zero-initialising them is safe.

Usage:
    python patch_vocab_for_spec_decode.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output ./models/Qwen2.5-0.5B-vocab152064 \
        --target-vocab 152064
"""
import argparse
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def download_model(model_id: str, cache_dir: str) -> str:
    print(f"Downloading {model_id} ...")
    return snapshot_download(model_id, cache_dir=cache_dir)


def patch(src_dir: str, dst_dir: str, target_vocab: int):
    os.makedirs(dst_dir, exist_ok=True)

    # Copy all non-weight files
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if fname.endswith(".safetensors"):
            continue
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, fname))

    # Patch config.json
    cfg_path = os.path.join(dst_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    original_vocab = cfg["vocab_size"]
    extra = target_vocab - original_vocab
    assert extra > 0, f"Target vocab {target_vocab} must be > source vocab {original_vocab}"
    cfg["vocab_size"] = target_vocab
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"config.json: vocab_size {original_vocab} → {target_vocab} (+{extra} padding tokens)")

    # Find and patch safetensors files
    st_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".safetensors"))
    assert st_files, f"No .safetensors files found in {src_dir}"

    # Determine if embeddings are tied (shared weight tensor)
    tied = cfg.get("tie_word_embeddings", False)
    print(f"tie_word_embeddings: {tied}")

    for st_file in st_files:
        src_st = os.path.join(src_dir, st_file)
        dst_st = os.path.join(dst_dir, st_file)
        print(f"Patching {st_file} ...")
        weights = load_file(src_st)
        patched = {}
        for key, tensor in weights.items():
            if key in ("model.embed_tokens.weight", "lm_head.weight"):
                if tensor.shape[0] == original_vocab:
                    pad = torch.zeros(extra, tensor.shape[1], dtype=tensor.dtype)
                    tensor = torch.cat([tensor, pad], dim=0)
                    print(f"  {key}: {original_vocab} → {tensor.shape[0]} rows")
            patched[key] = tensor
        save_file(patched, dst_st)

    print(f"\nPatched model saved to: {dst_dir}")
    print("Point vLLM speculative-config model to this local path.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="./models/Qwen2.5-0.5B-vocab152064")
    parser.add_argument("--target-vocab", type=int, default=152064)
    parser.add_argument("--cache-dir", default="./models/.cache")
    args = parser.parse_args()

    src_dir = download_model(args.model, args.cache_dir)
    patch(src_dir, args.output, args.target_vocab)


if __name__ == "__main__":
    main()
