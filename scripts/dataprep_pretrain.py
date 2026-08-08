"""
Pretraining Data Pipeline for Baby GLM-5.2
============================================

Streams, tokenizes, and writes 3.3B tokens to binary files for pretraining.

Data Sources:
  1. FineWeb-Edu       -- 2.0B tokens (high-quality web text)
  2. Wikipedia EN      -- 0.7B tokens (encyclopedic knowledge)
  3. Project Gutenberg -- 0.4B tokens (literary text)
  4. StarCoder Python  -- 0.2B tokens (code, optional)

Output:
  data/train.bin   -- ~3.267B tokens, binary uint16 memmap
  data/val.bin     -- ~0.033B tokens, binary uint16 memmap
  data/meta.json   -- tokenizer info, vocab size, token counts

Usage:
  python scripts/dataprep_pretrain.py                    # Full 3.3B token run
  python scripts/dataprep_pretrain.py --total_tokens 10000000  # Quick 10M test
  python scripts/dataprep_pretrain.py --no_code          # Skip code data

Estimated runtime: 2-4 hours on CPU with good internet (full run)
Estimated disk: ~7 GB for output .bin files
"""

import argparse
import json
import os
import time

import numpy as np
import tiktoken
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = "./data"
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_TOTAL_TOKENS = 3_300_000_000  # 3.3B tokens (Chinchilla+ for 120M params)
DEFAULT_VAL_RATIO = 0.01  # 1% validation split
DEFAULT_WRITE_CHUNK = 1_000_000  # Flush to disk every 1M tokens

# Data source allocation (fraction of total tokens)
SOURCE_ALLOCATIONS = {
    "fineweb_edu": 0.606,   # ~2.0B tokens
    "wikipedia": 0.212,     # ~0.7B tokens
    "gutenberg": 0.121,     # ~0.4B tokens
    "starcoder": 0.061,     # ~0.2B tokens (optional)
}

# End-of-text token for GPT-2 tokenizer
GPT2_EOT = 50256


# =============================================================================
# Dataset Iterators
# =============================================================================
# Each iterator yields raw text strings from a HuggingFace dataset stream.
# Streaming avoids holding the full dataset in memory.
# =============================================================================


def iter_fineweb_edu():
    """Stream high-quality educational web text from FineWeb-Edu."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for example in ds:
        text = example.get("text", "")
        if text:
            yield text


def iter_wikipedia():
    """Stream English Wikipedia articles (script-free wikimedia version)."""
    from datasets import load_dataset
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    for example in ds:
        text = example.get("text", "")
        if text:
            yield text


def iter_gutenberg():
    """Stream public domain books from Project Gutenberg (PG-19 subset)."""
    from datasets import load_dataset
    ds = load_dataset(
        "emozilla/pg19",
        split="train",
        streaming=True,
    )
    for example in ds:
        # pg19 uses "text" for the full book text, with "short_book_title" as metadata
        text = example.get("text", "")
        if text:
            yield text


def iter_starcoder():
    """
    Streams Python code from The Stack v2 (non-gated subset).

    To use the full StarCoder dataset instead, first authenticate:
        huggingface-cli login
    Then change 'bigcode/the-stack-v2-train-smol-ids' below to 'bigcode/starcoderdata'.
    """
    from datasets import load_dataset
    try:
        # Try the non-gated smol subset first
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
    except Exception:
        # Fallback: use codeparrot's cleaned Python dataset
        ds = load_dataset(
            "codeparrot/codeparrot-clean",
            split="train",
            streaming=True,
        )
        for example in ds:
            text = example.get("content", "")
            if text:
                yield text


# Map source names to their iterators
SOURCE_ITERATORS = {
    "fineweb_edu": iter_fineweb_edu,
    "wikipedia": iter_wikipedia,
    "gutenberg": iter_gutenberg,
    "starcoder": iter_starcoder,
}


# =============================================================================
# Tokenization & Writing
# =============================================================================


def tokenize_and_write(
    source_name,
    text_iterator,
    target_tokens,
    encoder,
    eot_token,
    train_file,
    val_file,
    val_ratio,
    write_chunk,
    dtype,
):
    """
    Tokenize text from an iterator and write tokens to train/val binary files.

    Each document is separated by an EOT token. Documents are randomly assigned
    to val split with probability val_ratio.

    Args:
        source_name: Name of the data source (for logging)
        text_iterator: Iterator yielding text strings
        target_tokens: Number of tokens to collect from this source
        encoder: tiktoken encoder
        eot_token: End-of-text token ID
        train_file: Open file handle for train.bin
        val_file: Open file handle for val.bin
        val_ratio: Fraction of documents for validation
        write_chunk: Buffer size before flushing to disk
        dtype: numpy dtype for token storage (uint16 or uint32)

    Returns:
        (train_tokens_written, val_tokens_written)
    """
    train_buffer = []
    val_buffer = []
    train_total = 0
    val_total = 0
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
        # Tokenize the document
        tokens = encoder.encode_ordinary(text)
        tokens.append(eot_token)  # Separate documents with EOT

        # Randomly assign entire documents to train or val
        is_val = rng.random() < val_ratio

        if is_val:
            val_buffer.extend(tokens)
            val_total += len(tokens)
            # Flush val buffer
            if len(val_buffer) >= write_chunk:
                val_file.write(np.array(val_buffer, dtype=dtype).tobytes())
                val_buffer = []
        else:
            train_buffer.extend(tokens)
            train_total += len(tokens)
            # Flush train buffer
            if len(train_buffer) >= write_chunk:
                train_file.write(np.array(train_buffer, dtype=dtype).tobytes())
                train_buffer = []

        doc_count += 1
        pbar.update(len(tokens))

        # Check if the target has been reached
        if train_total + val_total >= target_tokens:
            break

    # Flush remaining buffers
    if train_buffer:
        train_file.write(np.array(train_buffer, dtype=dtype).tobytes())
    if val_buffer:
        val_file.write(np.array(val_buffer, dtype=dtype).tobytes())

    pbar.close()
    print(f"    {source_name}: {doc_count:,} docs | "
          f"train {train_total:,} + val {val_total:,} = {train_total + val_total:,} tokens")

    return train_total, val_total


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Pretraining Data Pipeline for Baby GLM-5.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for .bin and .json files")
    parser.add_argument("--total_tokens", type=int, default=DEFAULT_TOTAL_TOKENS,
                        help="Total tokens to collect across all sources")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help="Fraction of documents for validation split")
    parser.add_argument("--write_chunk", type=int, default=DEFAULT_WRITE_CHUNK,
                        help="Buffer size (tokens) before flushing to disk")
    parser.add_argument("--no_code", action="store_true",
                        help="Exclude code data (StarCoder)")
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")
    meta_path = os.path.join(args.output_dir, "meta.json")

    # --- Tokenizer ---
    encoder = tiktoken.get_encoding(DEFAULT_TOKENIZER)
    vocab_size = encoder.n_vocab  # 50257 for GPT-2
    eot_token = GPT2_EOT

    # Determine dtype: uint16 if vocab fits, uint32 otherwise
    if vocab_size <= 65535:
        dtype = np.uint16
        dtype_str = "uint16"
    else:
        dtype = np.uint32
        dtype_str = "uint32"

    # --- Compute per-source token targets ---
    include_code = not args.no_code
    active_sources = {k: v for k, v in SOURCE_ALLOCATIONS.items()
                      if k != "starcoder" or include_code}

    # Renormalize allocations if code is excluded
    total_alloc = sum(active_sources.values())
    source_targets = {
        k: int(args.total_tokens * v / total_alloc)
        for k, v in active_sources.items()
    }

    # --- Print Plan ---
    print(f"\n{'='*70}")
    print(f"  Pretraining Data Pipeline for Baby GLM-5.2")
    print(f"{'='*70}")
    print(f"  Tokenizer:     {DEFAULT_TOKENIZER} (vocab_size={vocab_size})")
    print(f"  Token dtype:   {dtype_str}")
    print(f"  Total target:  {args.total_tokens:,} tokens")
    print(f"  Val ratio:     {args.val_ratio:.1%}")
    print(f"  Output dir:    {os.path.abspath(args.output_dir)}")
    print(f"  Code data:     {'Yes' if include_code else 'No'}")
    print(f"\n  Source Allocation:")
    for name, target in source_targets.items():
        print(f"    {name:20s}  {target:>14,} tokens  ({target/args.total_tokens:.1%})")
    print(f"{'='*70}\n")

    # --- Process Each Source ---
    total_train = 0
    total_val = 0
    t0 = time.time()

    with open(train_path, "wb") as train_file, open(val_path, "wb") as val_file:
        for source_name, target in source_targets.items():
            print(f"\n  [{source_name}] Streaming {target:,} tokens...")
            iterator_fn = SOURCE_ITERATORS[source_name]

            try:
                train_written, val_written = tokenize_and_write(
                    source_name=source_name,
                    text_iterator=iterator_fn(),
                    target_tokens=target,
                    encoder=encoder,
                    eot_token=eot_token,
                    train_file=train_file,
                    val_file=val_file,
                    val_ratio=args.val_ratio,
                    write_chunk=args.write_chunk,
                    dtype=dtype,
                )
                total_train += train_written
                total_val += val_written
            except Exception as e:
                print(f"  [WARN] Error streaming {source_name}: {e}")
                print(f"         Skipping this source and continuing...")
                continue

    elapsed = time.time() - t0

    # --- Write Metadata ---
    meta = {
        "tokenizer": DEFAULT_TOKENIZER,
        "vocab_size": vocab_size,
        "eot_token": eot_token,
        "dtype": dtype_str,
        "train_tokens": total_train,
        "val_tokens": total_val,
        "total_tokens": total_train + total_val,
        "sources": list(source_targets.keys()),
        "val_ratio": args.val_ratio,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- Summary ---
    train_size_gb = os.path.getsize(train_path) / 1e9
    val_size_gb = os.path.getsize(val_path) / 1e9

    print(f"\n{'='*70}")
    print(f"  [DONE] Data preparation complete!")
    print(f"{'='*70}")
    print(f"  Time elapsed:    {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print(f"  Train tokens:    {total_train:,}")
    print(f"  Val tokens:      {total_val:,}")
    print(f"  Total tokens:    {total_train + total_val:,}")
    print(f"  train.bin:       {train_size_gb:.2f} GB")
    print(f"  val.bin:         {val_size_gb:.2f} GB")
    print(f"  meta.json:       {meta_path}")
    print(f"\n  Next step: python train_glm5.py --data_dir {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
