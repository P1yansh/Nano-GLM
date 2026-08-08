---
title: Nano-GLM (120M MoE)
emoji: 🧪
colorFrom: indigo
colorTo: cyan
sdk: docker
app_file: app.py
pinned: false
---

# Nano-GLM (GLM-5.2 Baby 120M) - From Scratch

This repository contains a from-scratch implementation and pretraining script for a baby version (~120M parameters) of GLM-5.2 (GLM MoE DSA). The project is heavily inspired by Andrej Karpathy's nanoGPT and aims to serve as a highly educational resource.

The model is designed to be small enough to train on a single consumer laptop GPU (e.g., RTX 4050 6GB VRAM) while incorporating cutting-edge architectural innovations found in modern frontier models.

## Architectural Features Implemented

This implementation goes beyond a standard Transformer by incorporating three major innovations from recent frontier models (such as DeepSeek-V3 and GLM-5):

1. **MLA (Multi-Latent Attention):** Compresses the attention mechanism using LoRA-style projections to drastically reduce VRAM usage during training and inference.
2. **DSA (DeepSeek Sparse Attention):** Selects only the most relevant tokens to attend to via a learned indexer, rather than attending to the entire context uniformly.
3. **MoE (Mixture of Experts):** Employs a fine-grained sigmoid-routed mixture of experts alongside a shared expert, activating only a subset of parameters per token.

## Training Features

The training loop (train_glm5.py) is highly optimized for limited hardware (6GB VRAM) while maximizing throughput (achieving ~4,900 tokens/sec on an RTX 4050):
- **Mixed Precision:** Utilizes bfloat16 and TF32 Tensor Cores.
- **Gradient Checkpointing:** Recomputes forward passes during backpropagation to reduce VRAM consumption by approximately 40%.
- **Gradient Accumulation:** Enables large effective batch sizes on a single GPU.
- **WSD (Warmup-Stable-Decay) Learning Rate Schedule:** Supports multi-phase training by holding the learning rate at a peak for a stable exploration phase before initiating a steep cosine decay (controlled via the --stable_iters parameter).

## Repository Structure

```text
reproduce_glm5/
├── app.py                  # FastAPI backend server + static web UI hosting
├── train_glm5.py           # Core training loop, model architecture & sampling
├── export_model.py         # Utility to strip optimizer states for fast inference
├── requirements.txt        # Python dependencies
├── README.md               # Documentation & usage guide
├── docs/                   # Detailed documentation & guides
│   └── llm_training_guide.md
├── scripts/                # Data tokenization & dataset preparation scripts
│   ├── dataprep_pretrain.py
│   └── dataprep_phase3.py
├── static/                 # Production web frontend (HTML/CSS/JS)
│   └── index.html
├── data/                   # Tokenized dataset files (.bin / .json)
└── out_glm5/               # Model checkpoints & inference weights
```

## Educational Guide

For individuals new to LLM pretraining, learning rates, loss curves, and scaling laws, an included beginner guide is available:
[LLM Training Guide for Beginners](docs/llm_training_guide.md)

## Usage

### Data Preparation
```bash
# Tokenize and prepare pretraining dataset
python scripts/dataprep_pretrain.py
```

### Installation
```bash
pip install -r requirements.txt
```

### Training
To train the model on a single GPU using the WSD schedule (holding the learning rate stable for 217,000 steps), execute the following command:

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
To sample text from the best trained checkpoint:

```bash
python train_glm5.py --eval_only --ckpt out_glm5/ckpt_best.pt --prompt "The future of AI is"
```

## Model Exporting & Production API Deployment

### 1. Export Lightweight Inference Model
Training checkpoints contain optimizer states required for resuming (~1.4 GB). To prepare the model for cloud serving, strip the optimizer state down to an inference-only checkpoint (~240 MB):

```bash
python export_model.py --input_ckpt ckpt_best.pt --output_ckpt model_inference.pt
```

### 2. Launch FastAPI Server
Run the FastAPI production server using Uvicorn:

```bash
uvicorn app:app --reload --port 8000
```

The server automatically pre-loads the model into memory on startup and serves requests via CPU inference.

### 3. API Usage & Documentation
FastAPI provides an interactive OpenAPI / Swagger UI documentation page out of the box. Open `http://localhost:8000/docs` in a browser to test endpoints interactively.

#### Example REST API Call (`POST /generate`):

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "In conclusion,",
       "max_new_tokens": 100,
       "temperature": 0.8,
       "top_k": 50,
       "seed": 42
     }'
```

#### Example Response:
```json
{
  "prompt": "In conclusion,",
  "generated_text": "In conclusion, the development of modern neural network architectures...",
  "num_tokens": 103,
  "inference_time_sec": 1.42
}
```

## License
MIT License
