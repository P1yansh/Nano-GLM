# Nano-GLM (GLM-5.2 Baby 120M) — From Scratch

A from-scratch implementation and pretraining of a baby version (~120M parameters) of [GLM-5.2](https://github.com/THUDM/GLM-5) (GLM MoE DSA), trained entirely on a single consumer laptop GPU (RTX 4050, 6GB VRAM). Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Results

| Metric | Value |
|---|---|
| **Total Parameters** | 120,165,888 (tied) / 81.5M non-embedding |
| **Active Params per Token** | ~88.3M |
| **Training Data** | ~2.99 billion tokens (2.49B pretrain + 500M CPT) |
| **Final Validation Loss** | **2.6220 nats** (Perplexity 13.76) |
| **Training Throughput** | ~4,900 tokens/sec on RTX 4050 |
| **Inference Speed** | ~31.7 tokens/sec (CPU, single-token) |

> **Note:** Validation loss is measured on our own held-out validation split. For standardized evaluation, see the zero-shot benchmarks below.

### Standardized Zero-Shot Benchmarks

Evaluated on canonical base-model benchmarks following EleutherAI `lm-evaluation-harness` protocols (zero-shot, length-normalized conditional log-likelihood):

| Benchmark | Evaluation Task | Samples | Random Baseline | Nano-GLM Acc | Nano-GLM Acc (Norm) | Notes |
|---|---|---|---|---|---|---|
| **HellaSwag** | Commonsense reasoning (4 choices) | 1,000 | 25.0% | 31.8% | **32.7%** | Matches GPT-2 124M (~31–33%) |
| **ARC-Easy** | Grade-school science QA (4 choices) | 1,000 | 25.0% | 39.4% | **38.9%** | +13.9% over random chance |
| **LAMBADA** | Narrative next-word prediction | 1,000 | ~0.0% | 10.5% | **10.5%** | Strict exact match (PPL: 682) |

*Run benchmarks locally with: `python eval_benchmarks.py --all --limit 1000`*

### Sample Outputs (Step 322,000 Checkpoint)

**Prompt:** `"The French Revolution began in"` (Greedy, Rep=1.25, 4-Gram Block)
```
The French Revolution began in 1789, and it was a time of great change
for the country. The French had been trying to gain control over their
territories, but they were not able to do so. In 1801, the French
government took control of the region by taking control of all parts
of France. In 1804, the French army invaded Britain with the help of
the British forces at St. Louis...
```

**Prompt:** `def fibonacci(n):` (T=0.2, Min-P=0.08, Rep=1.05, N-Gram OFF)
```python
def fibonacci(n):
    """
    Compute the fibonacci sequence of a sequence of numbers.

    Examples
    --------
    >>> from numpy import *
    >>> n = np.array([1, 2, 3])
    >>> n.sort()
    [1, 2, 3]
```

> **About these outputs:** This is a **pre-trained base model** (not instruction-tuned or RLHF'd). It completes text by predicting the most likely next token based on its training data. The model correctly retrieves "1789" as the top-probability token for the French Revolution via greedy decoding, but may conflate related historical events (e.g., mixing in British/American references) because 1800s-era military conflicts are clustered nearby in the model's learned embedding space. This is expected behavior for a 120M-parameter base model — it has learned strong syntactic fluency and factual associations, but lacks the capacity for precise long-range factual consistency.

## Architecture

This implementation goes beyond a standard Transformer by incorporating three major innovations from DeepSeek-V3 and GLM-5:

### 1. Multi-Latent Attention (MLA)
Compresses the KV cache using LoRA-style low-rank projections. Instead of caching full keys and values (1,536 scalars/token/layer in standard MHA), MLA caches only **160 scalars/token/layer** — an **89.6% memory reduction**.

- Q LoRA rank: 384, KV LoRA rank: 128, shared RoPE key: 32-dim
- Interleaved RoPE positional encoding (base=10000.0)

### 2. DeepSeek Sparse Attention (DSA)
A learned indexer selects the **top-256 most relevant tokens** to attend to per query, rather than attending to the full context uniformly.

- Alternating Full/Sparse pattern across layers: `FSFSFSFSFSFS`
- Learned ReLU scoring (naturally zero-sparse, no masking needed)

### 3. Mixture of Experts (MoE)
Hybrid architecture: Layers 0–2 are dense SwiGLU, Layers 3–11 are MoE. Each MoE layer uses **8 routed experts (top-2 sigmoid routing) + 1 always-active shared expert**.

- **Sigmoid routing** (not softmax) — experts are scored independently, preventing expert collapse
- **Auxiliary-loss-free load balancing** via dynamic bias correction (`e_score_correction_bias`)
- Routed scaling factor: 2.5×

### Layer Structure
```
Layers 0-2:   [RMSNorm → MLA (Full Attn)  → RMSNorm → Dense SwiGLU MLP]
Layer 3:      [RMSNorm → MLA (Sparse Attn) → RMSNorm → MoE (8 Experts + Shared)]
Layers 4-11:  [RMSNorm → MLA (Alt F/S)     → RMSNorm → MoE (8 Experts + Shared)]
Final:        [RMSNorm → LM Head (weight-tied with embedding)]
```

## Decoding Engine

The `generate()` function applies the following techniques **in execution order**:

1. **Temperature Scaling** or **True Greedy Argmax** (T ≤ 0.01 bypasses sampling entirely)
2. **Multiplicative Repetition Penalty** (positive logits ÷ θ, negative × θ; default θ=1.2)
3. **No-Repeat N-Gram Blocking** (N=4 default; bans phrase repetition with -∞)
4. **Top-K Filtering** (K=200 CLI / K=50 web UI)
5. **Softmax Normalization**
6. **Min-P Truncation** (dynamic threshold = min_p × P_max; prunes low-confidence tokens)
7. **Multinomial Sampling**
8. **EOT Early Stopping** (breaks on token 50256 to prevent document boundary bleeding)

KV-cache provides ~67× speedup over full recomputation at each step.

## Training Features

Optimized for 6GB VRAM while maximizing throughput:

- **Mixed Precision:** bfloat16 with TF32 Tensor Cores
- **Gradient Checkpointing:** ~40% VRAM reduction (Layer 0 uncheckpointed for stability)
- **Gradient Accumulation:** Effective batch of 9,216 tokens/step
- **WSD Learning Rate Schedule:** Warmup → Stable → Cosine Decay, with `--warmup_start_iter` for Continued Pre-Training (CPT) without learning rate shock
- **Validation-Aware Checkpointing:** Saves both `ckpt.pt` (latest) and `ckpt_best.pt` (best val loss)

### Training Phases

| Phase | Steps | Data | Val Loss |
|---|---|---|---|
| Phase 1+2 (Pretrain) | 0 → 270,000 | 2.49B tokens (FineWeb-Edu, Wikipedia, Gutenberg, StarCoder) | 3.3734 |
| Phase 3 (CPT) | 270,000 → 324,300 | 500M tokens (Cosmopedia 40%, Python 30%, OpenWebMath 15%, FineWeb-Edu 15%) | **2.6220** |

## Repository Structure

```text
reproduce_glm5/
├── train_glm5.py           # Model architecture + training loop + generation (1524 lines)
├── app.py                  # FastAPI inference server (CPU default)
├── eval_benchmarks.py      # Standardized zero-shot benchmark suite (HellaSwag, ARC, LAMBADA)
├── eval_decoding.py        # 14-profile decoding ablation sweep
├── export_model.py         # Strip optimizer states for lightweight deployment
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── benchmark_results.md    # Standardized zero-shot benchmark scores
├── decoding_ablation_results.md  # Full ablation study (84 evaluations, 68 KB)
├── docs/
│   └── llm_training_guide.md     # Beginner-friendly training guide
├── scripts/
│   ├── dataprep_pretrain.py      # Phase 1+2 data tokenization
│   ├── dataprep_cpt.py           # Phase 3 CPT data pipeline
│   └── dataprep_phase3.py        # Alternative Phase 3 dataprep
├── static/
│   └── index.html                # Web UI with decoding parameter sliders
├── data/                         # Tokenized pretrain data (.bin)
├── data_cpt/                     # Tokenized CPT data
└── out_glm5/                     # Checkpoints (ckpt.pt, ckpt_best.pt, model_inference.pt)
```

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation
```bash
# Tokenize pretraining dataset
python scripts/dataprep_pretrain.py

# Tokenize CPT dataset (Phase 3)
python scripts/dataprep_cpt.py
```

### Training (Phase 1+2)
```bash
python train_glm5.py \
    --data_dir ./data \
    --batch_size 6 \
    --gradient_accumulation_steps 3 \
    --max_iters 270000 \
    --lr_decay_iters 260000 \
    --warmup_iters 1500 \
    --stable_iters 217000 \
    --eval_interval 2000 \
    --eval_iters 200 \
    --log_interval 100
```

### Continued Pre-Training (Phase 3)
```bash
python train_glm5.py \
    --data_dir ./data_cpt \
    --resume \
    --batch_size 6 \
    --gradient_accumulation_steps 3 \
    --learning_rate 1.8e-4 \
    --min_lr 3e-5 \
    --warmup_iters 1000 \
    --warmup_start_iter 270000 \
    --stable_iters 312000 \
    --lr_decay_iters 324300 \
    --eval_interval 2000 \
    --eval_iters 200 \
    --log_interval 100
```

### Text Generation
```bash
# Prose (greedy decoding, deterministic)
python train_glm5.py --eval_only --prompt "The French Revolution began in" \
    --temperature 0.0 --repetition_penalty 1.25 --no_repeat_ngram_size 4

# Code (near-greedy, no n-gram blocking)
python train_glm5.py --eval_only --prompt "def fibonacci(n):" \
    --temperature 0.2 --top_k 50 --repetition_penalty 1.05 --no_repeat_ngram_size 0
```

### Running Standard Benchmarks
Evaluate the model on zero-shot benchmarks (HellaSwag, ARC-Easy, LAMBADA):
```bash
# Run 1,000 samples per benchmark
python eval_benchmarks.py --all --limit 1000

# Or evaluate specific benchmarks:
python eval_benchmarks.py --benchmarks hellaswag arc_easy
```

## Deployment

### 1. Export Lightweight Model
Training checkpoints contain optimizer states (~1.4 GB). Strip them for deployment (~458 MB):
```bash
python export_model.py --input_ckpt ckpt_best.pt --output_ckpt model_inference.pt
```

### 2. Launch FastAPI Server
```bash
uvicorn app:app --reload --port 8000
```
The server pre-loads the model into memory on startup and serves requests via CPU inference (~31.7 tok/s).

### 3. API Usage
Interactive Swagger docs available at `http://localhost:8000/docs`.

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The French Revolution began in",
       "max_new_tokens": 150,
       "temperature": 0.3,
       "top_k": 50,
       "repetition_penalty": 1.2,
       "min_p": 0.08,
       "no_repeat_ngram_size": 4,
       "seed": 42
     }'
```

```json
{
  "prompt": "The French Revolution began in",
  "generated_text": "The French Revolution began in 1783, and it was a period of rapid industrialization that would shape the country. The revolution was driven by a desire to create a new government and a new way of life for the people. In 1801, the French Revolution took place in Paris...",
  "num_tokens": 150,
  "inference_time_sec": 4.73
}
```

## Educational Guide

For individuals new to LLM pretraining, loss curves, and scaling laws:
[LLM Training Guide for Beginners](docs/llm_training_guide.md)

## Hardware

Trained and tested on:
- **GPU:** NVIDIA GeForce RTX 4050 Laptop (6GB VRAM)
- **Laptop:** Lenovo LOQ 15 (15ARP9)
- **VRAM Usage:** ~4.78 GB during training (with gradient checkpointing)
- **Power Mode:** Performance Mode (Fn+Q, Red LED, 95W TGP) for full throughput

## License
MIT License
