"""
Standardized Zero-Shot Benchmark Suite for Nano-GLM (Baby GLM-5.2).

Evaluates the base pretrained model on canonical language modeling benchmarks:
  1. HellaSwag (Rowan et al.): Commonsense reasoning continuation (4 choices)
  2. ARC-Easy (Clark et al.): Grade-school science QA (4 choices)
  3. ARC-Challenge: Hard grade-school science QA (4 choices)
  4. LAMBADA (Paperno et al., OpenAI version): Narrative final-word prediction

Uses conditional log-likelihood scoring matching EleutherAI lm-evaluation-harness standards.
"""

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

# Allow unpickling checkpoints saved with __main__.GLM5Config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_glm5 import GLM5Config, GLM5ForCausalLM
sys.modules["__main__"].GLM5Config = GLM5Config


def load_model(ckpt_path, device):
    """Load GLM5 model from checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"  [>] Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GLM5ForCausalLM(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    step = checkpoint.get("iter_num", "N/A")
    val_loss = checkpoint.get("best_val_loss", "N/A")
    print(f"  [OK] Model loaded successfully (Step: {step}, Best Val Loss: {val_loss})")
    return model, config, step, val_loss


@torch.no_grad()
def score_multiple_choice_batch(model, enc, context: str, candidates: list, device, max_seq_len=512):
    """
    Score a single question with K candidate completions in a batched forward pass.
    
    Returns:
        raw_scores: list of sum log-probabilities for each candidate
        norm_scores: list of mean log-probabilities (length-normalized)
    """
    ctx_tokens = enc.encode(context)
    cand_tokens_list = [enc.encode(" " + c.strip()) if not c.startswith(" ") else enc.encode(c) for c in candidates]

    # Guard against sequences exceeding model block size
    seqs = []
    valid_cand_lens = []
    for c_tokens in cand_tokens_list:
        full = ctx_tokens + c_tokens
        if len(full) > max_seq_len:
            # Truncate context from the left to preserve completion
            overflow = len(full) - max_seq_len
            curr_ctx = ctx_tokens[overflow:]
            full = curr_ctx + c_tokens
            curr_ctx_len = len(curr_ctx)
        else:
            curr_ctx_len = len(ctx_tokens)
        seqs.append(full)
        valid_cand_lens.append((curr_ctx_len, len(c_tokens)))

    max_len = max(len(s) for s in seqs)
    padded = [s + [0] * (max_len - len(s)) for s in seqs]
    input_tensor = torch.tensor(padded, dtype=torch.long, device=device)

    # Forward pass
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.amp.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.no_grad()
    )

    with autocast_ctx:
        logits, _ = model(input_tensor)

    raw_scores = []
    norm_scores = []
    for i, (c_ctx_len, c_len) in enumerate(valid_cand_lens):
        if c_len == 0:
            raw_scores.append(-float("inf"))
            norm_scores.append(-float("inf"))
            continue

        c_tokens = torch.tensor(cand_tokens_list[i], dtype=torch.long, device=device)
        start_idx = c_ctx_len - 1
        end_idx = start_idx + c_len

        pred_logits = logits[i, start_idx:end_idx, :]
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        token_log_probs = log_probs[torch.arange(c_len, device=device), c_tokens]

        raw = token_log_probs.sum().item()
        norm = token_log_probs.mean().item()
        raw_scores.append(raw)
        norm_scores.append(norm)

    return raw_scores, norm_scores


def run_hellaswag(model, enc, device, limit=None):
    """Evaluate on HellaSwag validation split."""
    print("\n" + "=" * 70)
    print("  Evaluating HellaSwag (Commonsense Reasoning - 4 choices)")
    print("=" * 70)

    dataset = load_dataset("Rowan/hellaswag", split="validation")
    if limit is not None and limit < len(dataset):
        dataset = dataset.select(range(limit))

    num_samples = len(dataset)
    correct_raw = 0
    correct_norm = 0

    t0 = time.time()
    for i, item in enumerate(tqdm(dataset, desc="HellaSwag", total=num_samples)):
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        raw_scores, norm_scores = score_multiple_choice_batch(model, enc, ctx, endings, device)

        pred_raw = int(np.argmax(raw_scores))
        pred_norm = int(np.argmax(norm_scores))

        if pred_raw == label:
            correct_raw += 1
        if pred_norm == label:
            correct_norm += 1

    dt = time.time() - t0
    acc_raw = (correct_raw / num_samples) * 100.0
    acc_norm = (correct_norm / num_samples) * 100.0

    print(f"\n  [HellaSwag Results] ({num_samples:,} samples, {dt:.1f}s)")
    print(f"    Raw Accuracy:               {acc_raw:.2f}% (Random: 25.00%)")
    print(f"    Length-Normalized (acc_norm): {acc_norm:.2f}% (Standard metric)")

    return {
        "task": "HellaSwag",
        "num_samples": num_samples,
        "acc": acc_raw,
        "acc_norm": acc_norm,
        "random_baseline": 25.00,
        "time_sec": dt,
    }


def run_arc(model, enc, device, subset="ARC-Easy", split="test", limit=None):
    """Evaluate on AI2 ARC (Easy or Challenge)."""
    task_name = f"ARC ({subset})"
    print("\n" + "=" * 70)
    print(f"  Evaluating {task_name} - {split} split")
    print("=" * 70)

    dataset = load_dataset("ai2_arc", subset, split=split)
    if limit is not None and limit < len(dataset):
        dataset = dataset.select(range(limit))

    num_samples = len(dataset)
    correct_raw = 0
    correct_norm = 0

    t0 = time.time()
    for i, item in enumerate(tqdm(dataset, desc=task_name, total=num_samples)):
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        if answer_key in labels:
            label_idx = labels.index(answer_key)
        else:
            # Fallback for 1-based indexing e.g. '1', '2', '3', '4' vs 'A', 'B', 'C', 'D'
            letter_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
            mapped = letter_map.get(answer_key, answer_key)
            label_idx = labels.index(mapped) if mapped in labels else 0

        prompt = f"Question: {question}\nAnswer:"
        raw_scores, norm_scores = score_multiple_choice_batch(model, enc, prompt, choices, device)

        pred_raw = int(np.argmax(raw_scores))
        pred_norm = int(np.argmax(norm_scores))

        if pred_raw == label_idx:
            correct_raw += 1
        if pred_norm == label_idx:
            correct_norm += 1

    dt = time.time() - t0
    acc_raw = (correct_raw / num_samples) * 100.0
    acc_norm = (correct_norm / num_samples) * 100.0

    print(f"\n  [{task_name} Results] ({num_samples:,} samples, {dt:.1f}s)")
    print(f"    Raw Accuracy:               {acc_raw:.2f}% (Random: 25.00%)")
    print(f"    Length-Normalized (acc_norm): {acc_norm:.2f}% (Standard metric)")

    return {
        "task": task_name,
        "num_samples": num_samples,
        "acc": acc_raw,
        "acc_norm": acc_norm,
        "random_baseline": 25.00,
        "time_sec": dt,
    }


@torch.no_grad()
def run_lambada(model, enc, device, limit=None, max_seq_len=512):
    """Evaluate on LAMBADA OpenAI version (Exact Word Prediction & Perplexity)."""
    print("\n" + "=" * 70)
    print("  Evaluating LAMBADA (OpenAI) - Narrative Last Word Prediction")
    print("=" * 70)

    dataset = load_dataset("EleutherAI/lambada_openai", split="test")
    if limit is not None and limit < len(dataset):
        dataset = dataset.select(range(limit))

    num_samples = len(dataset)
    correct_word = 0
    total_log_likelihood = 0.0
    total_target_tokens = 0

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.amp.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.no_grad()
    )

    t0 = time.time()
    for i, item in enumerate(tqdm(dataset, desc="LAMBADA", total=num_samples)):
        text = item["text"]
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        context_str, target_word = parts
        target_str = " " + target_word

        ctx_tokens = enc.encode(context_str)
        target_tokens = enc.encode(target_str)
        if len(target_tokens) == 0:
            continue

        full_tokens = ctx_tokens + target_tokens
        if len(full_tokens) > max_seq_len:
            overflow = len(full_tokens) - max_seq_len
            full_tokens = full_tokens[overflow:]
            ctx_len = len(ctx_tokens) - overflow
        else:
            ctx_len = len(ctx_tokens)

        input_tensor = torch.tensor([full_tokens], dtype=torch.long, device=device)

        with autocast_ctx:
            logits, _ = model(input_tensor)

        # Check greedy prediction for target tokens
        c_len = len(target_tokens)
        start_idx = ctx_len - 1
        end_idx = start_idx + c_len

        pred_logits = logits[0, start_idx:end_idx, :]
        greedy_preds = torch.argmax(pred_logits, dim=-1).tolist()

        if greedy_preds == target_tokens:
            correct_word += 1

        # Calculate perplexity over target tokens
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=device)
        token_nll = -log_probs[torch.arange(c_len, device=device), target_tensor].sum().item()

        total_log_likelihood += token_nll
        total_target_tokens += c_len

    dt = time.time() - t0
    acc = (correct_word / num_samples) * 100.0
    ppl = np.exp(total_log_likelihood / max(1, total_target_tokens))

    print(f"\n  [LAMBADA Results] ({num_samples:,} samples, {dt:.1f}s)")
    print(f"    Accuracy:                   {acc:.2f}% (Random: ~0.00%)")
    print(f"    Perplexity:                 {ppl:.2f}")

    return {
        "task": "LAMBADA (OpenAI)",
        "num_samples": num_samples,
        "acc": acc,
        "acc_norm": acc,
        "perplexity": ppl,
        "random_baseline": 0.00,
        "time_sec": dt,
    }


def save_markdown_report(results, model_step, model_val_loss, output_path):
    """Save formatted markdown summary report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Nano-GLM Standardized Benchmark Evaluation Report",
        "",
        f"**Date:** {now}  ",
        f"**Model Checkpoint:** Step {model_step:,} (Val Loss: {model_val_loss})  ",
        f"**Architecture:** GLM-5.2 Baby (120M params, MLA + DSA + MoE)  ",
        "",
        "## Summary Table",
        "",
        "| Benchmark | Evaluation Task | Samples | Random Baseline | Nano-GLM Acc | Nano-GLM Acc (Norm) | Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        task = r["task"]
        n = r["num_samples"]
        rand = f"{r['random_baseline']:.1f}%"
        acc = f"{r['acc']:.2f}%"
        acc_norm = f"{r['acc_norm']:.2f}%"
        extra = f"Perplexity: {r['perplexity']:.2f}" if "perplexity" in r else "Normalized conditional LL"
        if task == "HellaSwag":
            desc = "Commonsense reasoning (4 choices)"
        elif "ARC" in task:
            desc = "Science reasoning QA (4 choices)"
        else:
            desc = "Narrative word prediction"
        lines.append(f"| **{task}** | {desc} | {n:,} | {rand} | {acc} | **{acc_norm}** | {extra} |")

    lines.extend([
        "",
        "## Methodology Notes",
        "- All evaluations were conducted in **zero-shot** mode matching standard EleutherAI `lm-evaluation-harness` protocols.",
        "- For multiple-choice questions, **Acc (Norm)** is length-normalized: $\\frac{1}{K} \\sum_{j=1}^K \\log P(t_j \\mid \\text{context}, t_{<j})$, which prevents scoring bias toward shorter answers.",
        "- LAMBADA tests strict exact-match prediction on the final target word of narrative excerpts.",
        "",
        "---",
        "*Report generated automatically by `eval_benchmarks.py`.*",
    ])

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  [OK] Saved markdown report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nano-GLM on Standard Base Benchmarks")
    parser.add_argument("--ckpt", type=str, default="out_glm5/ckpt_best.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        choices=["all", "hellaswag", "arc_easy", "arc_challenge", "lambada"],
        help="Benchmarks to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="Sample limit per task (default: full test/val set)")
    parser.add_argument("--output", type=str, default="benchmark_results.md", help="Output markdown report path")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"  [>] Using evaluation device: {device}")

    model, config, step, val_loss = load_model(args.ckpt, device)
    enc = tiktoken.get_encoding("gpt2")

    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["hellaswag", "arc_easy", "arc_challenge", "lambada"]

    results = []
    for b in benchmarks:
        if b == "hellaswag":
            res = run_hellaswag(model, enc, device, limit=args.limit)
            results.append(res)
        elif b == "arc_easy":
            res = run_arc(model, enc, device, subset="ARC-Easy", split="test", limit=args.limit)
            results.append(res)
        elif b == "arc_challenge":
            res = run_arc(model, enc, device, subset="ARC-Challenge", split="test", limit=args.limit)
            results.append(res)
        elif b == "lambada":
            res = run_lambada(model, enc, device, limit=args.limit)
            results.append(res)

    save_markdown_report(results, step, val_loss, args.output)


if __name__ == "__main__":
    main()
