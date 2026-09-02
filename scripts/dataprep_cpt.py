"""
Pretraining Data Pipeline -- Continued Pre-Training (CPT) (0.5B Tokens Target)
===============================================================================

Streams 500 Million tokens of high-density pedagogical and reasoning data:
  1. Cosmopedia v2 (HuggingFaceTB/cosmopedia-v2)  -- 200M tokens (40%) [Synthetic Textbooks]
  2. Python Code (The Stack / CodeParrot)          -- 150M tokens (30%) [Syntax & Scope Logic]
  3. OpenWebMath (open-web-math/open-web-math)     --  75M tokens (15%) [Strict Mathematical Reasoning]
  4. FineWeb-Edu Score 4+ (sample-10BT)            --  75M tokens (15%) [Top-Tier Filtered Web Knowledge]

Outputs to `./data_cpt` with `train.npy`, `val.npy`, and `meta.json`.

Usage:
  python scripts/dataprep_cpt.py
  python scripts/dataprep_cpt.py --total_tokens 500000000
"""

import argparse
import json
import os
import time

import numpy as np
import tiktoken
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = "./data_cpt"
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_TOTAL_TOKENS = 500_000_000  # 0.5B tokens
DEFAULT_VAL_RATIO = 0.01
DEFAULT_WRITE_CHUNK = 1_000_000

SOURCE_ALLOCATIONS = {
    "cosmopedia": 0.40,    # 200M tokens
    "python_code": 0.30,   # 150M tokens
    "openwebmath": 0.15,   #  75M tokens
    "fineweb_edu_top": 0.15, #  75M tokens
}

GPT2_EOT = 50256


def iter_cosmopedia():
    """Stream synthetic textbook data from Cosmopedia v2."""
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)
        for example in ds:
            text = example.get("text", "")
            if text:
                yield text
    except Exception as e:
        print(f"  [WARN] Cosmopedia v2 fallback to v1 due to: {e}")
        ds = load_dataset("HuggingFaceTB/cosmopedia", "web_samples", split="train", streaming=True)
        for example in ds:
            text = example.get("text", "")
            if text:
                yield text


def iter_python_code():
    """Stream clean Python code from The Stack smol or CodeParrot."""
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            streaming=True,
        )
        for example in ds:
            text = example.get("content", "")
            if text:
                yield text
    except Exception as e:
        print(f"  [WARN] Falling back to codeparrot-clean due to: {e}")
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        for example in ds:
            text = example.get("content", "")
            if text:
                yield text


def iter_openwebmath():
    """Stream mathematical reasoning data from OpenWebMath."""
    from datasets import load_dataset
    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        for example in ds:
            text = example.get("text", "")
            if text:
                yield text
    except Exception as e:
        print(f"  [WARN] open-web-math stream failed: {e}. Falling back to Wikipedia...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for example in ds:
            text = example.get("text", "")
            if text:
                yield text


def iter_fineweb_edu_top():
    """Stream top educational tier (score >= 4) from FineWeb-Edu."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for example in ds:
        score = example.get("int_score", None)
        if score is None:
            score = example.get("score", 0)
        # Only keep top-tier educational texts
        if score >= 4:
            text = example.get("text", "")
            if text:
                yield text


SOURCE_ITERATORS = {
    "cosmopedia": iter_cosmopedia,
    "python_code": iter_python_code,
    "openwebmath": iter_openwebmath,
    "fineweb_edu_top": iter_fineweb_edu_top,
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
    parser = argparse.ArgumentParser(description="Pretraining Data Pipeline -- Continued Pre-Training (CPT)")
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

    print(f"\n{'='*75}")
    print(f"  Continued Pre-Training (CPT) Data Pipeline (0.5B Tokens Target)")
    print(f"{'='*75}")
    print(f"  Output Dir: {os.path.abspath(args.output_dir)}")
    for name, target in source_targets.items():
        print(f"    {name:20s}: {target:,} tokens ({SOURCE_ALLOCATIONS[name]*100:.0f}%)")
    print(f"{'='*75}\n")

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
                print(f"  [ERROR] Failed stream on {source_name}: {e}")

    meta = {
        "tokenizer": DEFAULT_TOKENIZER,
        "vocab_size": vocab_size,
        "eot_token": GPT2_EOT,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "train_tokens": total_train,
        "val_tokens": total_val,
        "total_tokens": total_train + total_val,
        "sources": list(source_targets.keys()),
        "val_ratio": DEFAULT_VAL_RATIO,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  [DONE] CPT data ready at {args.output_dir} in {elapsed/60:.1f} mins!")
    print(f"         Total Tokens: {total_train + total_val:,} (Train: {total_train:,}, Val: {total_val:,})")


if __name__ == "__main__":
    main()
