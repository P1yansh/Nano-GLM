# 🍼 Nano-GLM (GLM-5.2 Baby 120M) - From Scratch

This repository contains a from-scratch implementation and pretraining script for a "baby" version (~120M parameters) of **GLM-5.2** (GLM MoE DSA). It is heavily inspired by Andrej Karpathy's `nanoGPT` and aims to be highly educational.

The model is small enough to train on a single consumer laptop GPU (e.g., RTX 4050 6GB VRAM) but includes all the cutting-edge architectural innovations of modern frontier models.

## ✨ Architectural Features Implemented

This isn't just a standard Transformer. It implements three major innovations from recent frontier models (like DeepSeek-V3 and GLM-5):

1. **MLA (Multi-Latent Attention):** Compresses the attention mechanism using LoRA-style projections to drastically save VRAM during training and inference.
2. **DSA (DeepSeek Sparse Attention):** Selects only the most relevant tokens to attend to via a learned indexer, rather than attending to the entire context uniformly.
3. **MoE (Mixture of Experts):** Uses a fine-grained sigmoid-routed mixture of experts alongside a shared expert, activating only a subset of parameters per token.

## 🚀 Training Features

The training loop (`train_glm5.py`) is highly optimized for limited hardware (6GB VRAM) while maximizing throughput (~4,900 tokens/sec on an RTX 4050):
- **Mixed Precision:** Uses `bfloat16` and TF32 Tensor Cores.
- **Gradient Checkpointing:** Recomputes forward passes during backprop to save ~40% VRAM.
- **Gradient Accumulation:** Achieves large effective batch sizes on a single GPU.
- **WSD (Warmup-Stable-Decay) Learning Rate Schedule:** 
  Supports multi-phase training by holding the learning rate at peak for a "stable" exploration phase before initiating a steep cosine decay. (Controlled via `--stable_iters`).

## 📚 Educational Guide

If you are new to LLM pretraining, learning rates, loss curves, and scaling laws, check out the included beginner guide:
👉 **[LLM Training Guide for Beginners](llm_training_guide.md)**

## 🛠️ Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
To train the model on a single GPU with the WSD schedule (holding LR stable for 217,000 steps):

```bash
python train_glm5.py \
    --data_dir ./data \
    --batch_size 6 \
    --gradient_accumulation_steps 3 \
    --max_iters 110000 \
    --lr_decay_iters 260000 \
    --warmup_iters 1500 \
    --stable_iters 217000 \
    --eval_interval 2000 \
    --eval_iters 200 \
    --log_interval 100
```

### Generation / Sampling
To sample text from your best trained checkpoint:

```bash
python train_glm5.py --eval_only --ckpt out_glm5/ckpt_best.pt --prompt "The future of AI is"
```

## ⚖️ License
MIT License
