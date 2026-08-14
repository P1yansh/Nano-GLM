"""
Pretraining Data Pipeline -- Phase 3 (0.4B Tokens)
===================================================

Streams 400 Million tokens of domain-diverse text:
  - Wikipedia (wikimedia/wikipedia)  -- 200M tokens (50%)
  - Gutenberg (emozilla/pg19)        -- 120M tokens (30%)
  - Python Code (bigcode/the-stack)  --  80M tokens (20%)

Outputs to `./data_phase3` to avoid touching active training in `./data`.

Usage:
  python scripts/dataprep_phase3.py
"""

import argparse
import json
import os
import time

import numpy as np
import tiktoken
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = "./data_phase3"
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_TOTAL_TOKENS = 400_000_000  # 0.4B tokens
DEFAULT_VAL_RATIO = 0.01
DEFAULT_WRITE_CHUNK = 1_000_000

SOURCE_ALLOCATIONS = {
    "wikipedia": 0.50,  # 200M tokens
    "gutenberg": 0.30,  # 120M tokens
    "starcoder": 0.20,  # 80M tokens
}

GPT2_EOT = 50256


def iter_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for example in ds:
        text = example.get("text", "")
        if text:
            yield text


def iter_gutenberg():
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    for example in ds:
        text = example.get("text", "")
        if text:
            yield text


def iter_starcoder():
    from datasets import load_dataset
    try:
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        for example in ds:
            text = example.get("content", "")
            if text:
                yield text
    except Exception:
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        for example in ds:
            text = example.get("content", "")
            if text:
                yield text


SOURCE_ITERATORS = {
    "wikipedia": iter_wikipedia,
    "gutenberg": iter_gutenberg,
    "starcoder": iter_starcoder,
}


def tokenize_and_write(source_name, text_iterator, target_tokens, encoder, eot_token, train_file, val_file, val_ratio, write_chunk, dtype):
    train_buffer, val_buffer = [], []
    train_total, val_total = 0, 0
    doc_count = 0
    rng = np.random.default_rng(seed=42 + hash(source_name) % 10000)

    pbar = tqdm(
        total=target_tokens,
        unit="tok",
        unit_scale=True,
        desc=f"  {source_name}",
        bar_format="  {desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for text in text_iterator:
        tokens = encoder.encode_ordinary(text)
        tokens.append(eot_token)

        is_val = rng.random() < val_ratio

        if is_val:
            val_buffer.extend(tokens)
            val_total += len(tokens)
            if len(val_buffer) >= write_chunk:
                val_file.write(np.array(val_buffer, dtype=dtype).tobytes())
                val_buffer = []
        else:
            train_buffer.extend(tokens)
            train_total += len(tokens)
            if len(train_buffer) >= write_chunk:
                train_file.write(np.array(train_buffer, dtype=dtype).tobytes())
                train_buffer = []

        doc_count += 1
        pbar.update(len(tokens))

        if train_total + val_total >= target_tokens:
            break

    if train_buffer:
        train_file.write(np.array(train_buffer, dtype=dtype).tobytes())
    if val_buffer:
        val_file.write(np.array(val_buffer, dtype=dtype).tobytes())

    pbar.close()
    print(f"    {source_name}: {doc_count:,} docs | train {train_total:,} + val {val_total:,} tokens")
    return train_total, val_total


def main():
    parser = argparse.ArgumentParser(description="Pretraining Data Pipeline -- Phase 3")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total_tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.npy")
    val_path = os.path.join(args.output_dir, "val.npy")
    meta_path = os.path.join(args.output_dir, "meta.json")

    encoder = tiktoken.get_encoding(DEFAULT_TOKENIZER)
    vocab_size = encoder.n_vocab
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    source_targets = {k: int(args.total_tokens * v) for k, v in SOURCE_ALLOCATIONS.items()}

    print(f"\n{'='*70}")
    print(f"  Phase 3 Data Pipeline (0.4B Tokens Target)")
    print(f"{'='*70}")
    print(f"  Output Dir: {os.path.abspath(args.output_dir)}")
    for name, target in source_targets.items():
        print(f"    {name:15s}: {target:,} tokens")
    print(f"{'='*70}\n")

    total_train, total_val = 0, 0
    t0 = time.time()

    with open(train_path, "wb") as train_file, open(val_path, "wb") as val_file:
        for source_name, target in source_targets.items():
            print(f"\n  [{source_name}] Streaming {target:,} tokens...")
            try:
                tr, va = tokenize_and_write(
                    source_name, SOURCE_ITERATORS[source_name](), target,
                    encoder, GPT2_EOT, train_file, val_file, DEFAULT_VAL_RATIO,
                    DEFAULT_WRITE_CHUNK, dtype
                )
                total_train += tr
                total_val += va
            except Exception as e:
                print(f"  [WARN] Error on {source_name}: {e}")

    meta = {
        "tokenizer": DEFAULT_TOKENIZER,
        "vocab_size": vocab_size,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "train_tokens": total_train,
        "val_tokens": total_val,
        "total_tokens": total_train + total_val,
        "sources": list(source_targets.keys()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  [DONE] Phase 3 data ready at {args.output_dir}! Total: {total_train + total_val:,} tokens")


if __name__ == "__main__":
    main()
