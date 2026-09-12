# Nano-GLM Standardized Benchmark Evaluation Report

**Date:** 2026-09-13 01:35:07  
**Model Checkpoint:** Step 322,000 (Val Loss: 2.6220385819673537)  
**Architecture:** GLM-5.2 Baby (120M params, MLA + DSA + MoE)  

## Summary Table

| Benchmark | Evaluation Task | Samples | Random Baseline | Nano-GLM Acc | Nano-GLM Acc (Norm) | Notes |
|---|---|---|---|---|---|---|
| **HellaSwag** | Commonsense reasoning (4 choices) | 1,000 | 25.0% | 31.80% | **32.70%** | Normalized conditional LL |
| **ARC (ARC-Easy)** | Science reasoning QA (4 choices) | 1,000 | 25.0% | 39.40% | **38.90%** | Normalized conditional LL |
| **LAMBADA (OpenAI)** | Narrative word prediction | 1,000 | 0.0% | 10.50% | **10.50%** | Perplexity: 682.14 |

## Methodology Notes
- All evaluations were conducted in **zero-shot** mode matching standard EleutherAI `lm-evaluation-harness` protocols.
- For multiple-choice questions, **Acc (Norm)** is length-normalized: $\frac{1}{K} \sum_{j=1}^K \log P(t_j \mid \text{context}, t_{<j})$, which prevents scoring bias toward shorter answers.
- LAMBADA tests strict exact-match prediction on the final target word of narrative excerpts.

---
*Report generated automatically by `eval_benchmarks.py`.*