"""
Let's Reproduce GLM-5.2 (GLM MoE DSA) — From Scratch!
=======================================================

A baby version of GLM-5.2 (zhipu-ai / zai-org / GLM-5), trained from scratch.

GLM-5.2 combines three cutting-edge innovations:
  1. MLA (Multi-Latent Attention) — LoRA-compressed Q and KV projections
  2. DSA (DeepSeek Sparse Attention) — top-k token selection via a learned indexer
  3. MoE (Mixture of Experts) — sigmoid-routed fine-grained experts + shared expert

This script implements ALL of these from scratch in a single file,
scaled down to ~120M parameters for training on a single consumer GPU.

Architecture Reference:
  HuggingFace transformers — models/glm_moe_dsa/modeling_glm_moe_dsa.py

Paper References:
  - DeepSeek-V3 (MLA + MoE): https://arxiv.org/abs/2412.19437
  - DeepSeek Sparse Attention: https://arxiv.org/abs/2603.12201
  - GLM: https://github.com/THUDM/GLM

Inspired by Andrej Karpathy's "Let's reproduce GPT-2" and nanoGPT.

Usage:
    python train_glm5.py                                # Train with defaults (RTX 4050 friendly)
    python train_glm5.py --batch_size 2                  # Smaller batch for less VRAM
    python train_glm5.py --compile                       # Use torch.compile (faster, needs warmup)
    python train_glm5.py --eval_only --ckpt out/ckpt.pt  # Generate from a checkpoint
    python train_glm5.py --no_gradient_checkpointing     # Disable grad checkpointing (needs more VRAM)
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# =============================================================================
# Section 1: Model Configuration
# =============================================================================
# The full GLM-5.2 has 78 layers, 6144 hidden, 256 experts — far too large.
# Everything is scaled down to ~120M total params while preserving EVERY
# architectural innovation. Think of this as "baby GLM-5.2".
# =============================================================================


@dataclass
class GLM5Config:
    """
    Configuration for baby GLM-5.2.
    Default values give a ~159M total param model (~82M non-embedding),
    trainable on a single RTX 4050 (6GB VRAM) with gradient checkpointing + bf16.

    The full GLM-5.2 values are shown in comments for reference.
    """

    # --- Vocabulary & Embedding ---
    vocab_size: int = 50304  # GPT-2 tokenizer (50257) padded to nearest 128  [full: 154880]

    # --- Core Dimensions ---
    hidden_size: int = 768  # Model width (d_model)                           [full: 6144]
    num_hidden_layers: int = 12  # Total decoder layers                       [full: 78]

    # --- Multi-Latent Attention (MLA) ---
    num_attention_heads: int = 12  # Number of query heads                    [full: 64]
    q_lora_rank: int = 384  # Query LoRA bottleneck                           [full: 2048]
    kv_lora_rank: int = 128  # Key/Value LoRA bottleneck                      [full: 512]
    qk_nope_head_dim: int = 32  # Non-rotary Q/K head dim                    [full: 192]
    qk_rope_head_dim: int = 32  # Rotary Q/K head dim                        [full: 64]
    v_head_dim: int = 64  # Value head dim                                    [full: 256]

    # --- Dense MLP ---
    intermediate_size: int = 2048  # Dense FFN intermediate dim               [full: 12288]
    hidden_act: str = "silu"  # Activation function

    # --- Mixture of Experts (MoE) ---
    first_k_dense_replace: int = 3  # First K layers use dense MLP           [full: 3]
    moe_intermediate_size: int = 256  # Per-expert FFN intermediate dim       [full: 2048]
    n_routed_experts: int = 8  # Number of routed experts                     [full: 256]
    num_experts_per_tok: int = 2  # Top-k experts per token                   [full: 8]
    n_shared_experts: int = 1  # Always-active shared experts                 [full: 1]
    n_group: int = 1  # Expert groups for routing                             [full: 1]
    topk_group: int = 1  # Top groups selected                                [full: 1]
    routed_scaling_factor: float = 2.5  # Expert weight scaling               [full: 2.5]
    norm_topk_prob: bool = True  # Normalize routing probabilities

    # --- DeepSeek Sparse Attention (DSA) ---
    index_topk: int = 256  # Top tokens selected by DSA indexer               [full: 2048]
    index_head_dim: int = 64  # Head dim in DSA indexer                       [full: 128]
    index_n_heads: int = 12  # Heads in DSA indexer                           [full: 32]

    # --- Positional Encoding ---
    max_position_embeddings: int = 4096  # Max context length                 [full: 202752]
    rope_theta: float = 10000.0  # RoPE base frequency

    # --- Regularization & Precision ---
    rms_norm_eps: float = 1e-5  # RMSNorm epsilon
    attention_dropout: float = 0.0  # Attention dropout

    # --- Initialization ---
    initializer_range: float = 0.02  # Std dev for weight init

    # --- Weight Tying ---
    # if False, Total model param = 158.8 M params. If True param cout changes to around 120 M
    tie_word_embeddings: bool = True  # Tie embed + lm_head                  [full: False]

    def __post_init__(self):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # Per-layer MLP type: first K layers = dense, rest = MoE (sparse)
        n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
        self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.num_hidden_layers - n_dense)
        # DSA indexer pattern: alternating "full" (run indexer) / "shared" (reuse previous)
        # Full GLM-5.2 uses a freq/offset schedule; this simplifies to alternating.
        self.indexer_types = [
            "full" if i % 2 == 0 else "shared" for i in range(self.num_hidden_layers)
        ]


# =============================================================================
# Section 2: Architecture Components
# =============================================================================
# Each component is built bottom-up, with detailed comments explaining
# WHY each design choice was made in GLM-5.2.
# =============================================================================

# ---------------------------------------------------------------------------
# 2a: RMSNorm — Root Mean Square Layer Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization).

    Unlike LayerNorm, RMSNorm does NOT center activations (no mean subtraction).
    This is cheaper and works just as well for LLMs.

    Formula: output = x / sqrt(mean(x²) + eps) * weight
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()  # Always compute in float32 for numerical stability
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# 2b: Rotary Position Embedding (RoPE) — Interleaved variant
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """
    Standard Rotary Position Embedding (RoPE).

    Computes cos/sin tables for position encoding. The actual rotation is
    applied by apply_rotary_pos_emb_interleave() — see below.
    """

    def __init__(self, dim, max_position_embeddings=4096, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: [B, T, D] — only used for dtype/device reference
            position_ids: [B, T]
        Returns:
            cos, sin: each [B, T, dim]
        """
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq.float() @ pos.float()).transpose(1, 2)  # [B, T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply INTERLEAVED Rotary Position Embedding.

    GLM-5.2 (and DeepSeek) uses interleaved RoPE pairs: (x0,x1), (x2,x3), ...
    Each pair is rotated by a single frequency.

    This is DIFFERENT from standard LLaMA-style RoPE which splits the
    first/second half of the head dimension. The interleaved version avoids
    memory-shuffling copies from 'rotate_half'.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Standard RoPE:     [x0..x_d/2 | x_d/2..x_d] → rotate halves  │
    │  Interleaved RoPE:  [x0,x1 | x2,x3 | ...] → rotate pairs      │
    └──────────────────────────────────────────────────────────────────┘
    """
    # cos/sin come as cat(freqs, freqs) → take the first half
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)

    # Split into even and odd indexed elements (the interleaved pairs)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]

    # Apply 2D rotation to each (even, odd) pair
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# 2c: Gated MLP (SwiGLU)
# ---------------------------------------------------------------------------


class GatedMLP(nn.Module):
    """
    SwiGLU MLP: down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

    The gating mechanism (SiLU on gate, element-wise multiply with up)
    consistently outperforms vanilla ReLU/GELU FFNs in modern LLMs.
    Used in LLaMA, DeepSeek, GLM, Gemma, Qwen, and many others.
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# 2d: Top-K Expert Router (Sigmoid-based, DeepSeek-style)
# ---------------------------------------------------------------------------


class TopKRouter(nn.Module):
    """
    Sigmoid-based top-k expert router with bias correction.

    Key differences from the traditional softmax MoE router:
    ┌──────────────────────────────────────────────────────────────────┐
    │  1. SIGMOID scoring (not softmax) — experts scored independently │
    │  2. Correction bias — loaded from checkpoint, helps balance load │
    │  3. Group routing — select top groups, then experts within them  │
    │  4. Normalize + scale — weights normalized then scaled by 2.5x  │
    └──────────────────────────────────────────────────────────────────┘

    The sigmoid approach prevents "expert collapse" where softmax routing
    causes only a few experts to receive all the tokens.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        # Router weight: one logit per expert
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        # Correction bias: pretrained load-balancing signal (zeros for training from scratch)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, x):
        x_flat = x.view(-1, self.hidden_dim)

        # Step 1: Sigmoid scoring (NOT softmax!)
        # Each expert gets an independent 0-1 probability
        router_logits = F.linear(x_flat.float(), self.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias

        # Step 2: Group-based routing
        # With n_group=1 (the current config), this is standard top-k.
        # With n_group>1 (full GLM-5.2), first select best groups, then pick
        # top experts only from those groups — prevents cross-group interference.
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.num_experts // self.n_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

        # Step 3: Select top-k experts per token
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)

        # Step 4: Normalize probabilities and scale
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_indices


# ---------------------------------------------------------------------------
# 2e: MoE Expert Collection (Batched 3D Tensors)
# ---------------------------------------------------------------------------


class MoEExperts(nn.Module):
    """
    Collection of expert MLPs stored as batched 3D parameter tensors.

    Instead of N separate nn.Linear modules, ALL expert weights are stored
    in single tensors. This enables efficient batched dispatch.

      gate_up_proj: [num_experts, 2*intermediate, hidden]
      down_proj:    [num_experts, hidden, intermediate]

    Each expert computes: SiLU(gate(x)) * up(x) → down → output

    NOTE: This is the naive loop implementation. Production systems (DeepSeek,
    GLM-5.2) use custom CUDA kernels for grouped GEMM — orders of magnitude faster.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # Fused gate+up projection: [E, 2*I, D]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        # Down projection: [E, D, I]
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

    def forward(self, x, topk_indices, topk_weights):
        """
        Route tokens to their selected experts and accumulate outputs.

        Args:
            x: [num_tokens, hidden_dim]
            topk_indices: [num_tokens, top_k] — which experts each token uses
            topk_weights: [num_tokens, top_k] — routing weights
        """
        final = torch.zeros_like(x)

        # Build per-expert assignment mask
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, self.num_experts)  # [tokens, top_k, E]
            expert_mask = expert_mask.permute(2, 1, 0)  # [E, top_k, tokens]
            expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

        # Process each active expert
        for idx in expert_hit:
            e = idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current = x[token_idx]

            # SwiGLU: SiLU(gate) * up → down
            gate, up = F.linear(current, self.gate_up_proj[e]).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            out = F.linear(hidden, self.down_proj[e])

            # Weight by routing probability and accumulate
            out = out * topk_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        return final


# ---------------------------------------------------------------------------
# 2f: Full MoE Block (Router + Routed Experts + Shared Expert)
# ---------------------------------------------------------------------------


class MoEBlock(nn.Module):
    """
    Full Mixture-of-Experts block.

    Output = Routed_Experts(x) + Shared_Expert(x)

    The shared expert ALWAYS processes all tokens — it provides a stable
    "backbone" of computation. The routed experts add specialized capacity
    for different types of tokens/patterns.
    """

    def __init__(self, config):
        super().__init__()
        self.gate = TopKRouter(config)
        self.experts = MoEExperts(config)
        # Shared expert: always-on, processes every token unconditionally
        self.shared_experts = GatedMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
        )

    def forward(self, x):
        residual = x
        orig_shape = x.shape
        topk_weights, topk_indices = self.gate(x)
        x = x.view(-1, x.shape[-1])
        x = self.experts(x, topk_indices, topk_weights).view(*orig_shape)
        x = x + self.shared_experts(residual)
        return x


# ---------------------------------------------------------------------------
# 2g: DeepSeek Sparse Attention (DSA) Indexer
# ---------------------------------------------------------------------------


class DSAIndexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) Indexer.

    THE key innovation of DSA: instead of attending to ALL past tokens (O(n²)),
    the indexer selects only the top-k most relevant tokens per query position.
    This makes long-context attention tractable (O(n·k) where k << n).

    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. Separate Q/K projections (NOT shared with main MLA attention)  │
    │  2. Multi-head dot-product scoring with ReLU (not softmax!)        │
    │  3. Learned per-head importance weights for aggregation            │
    │  4. Returns top-k token indices for the main attention to use      │
    └─────────────────────────────────────────────────────────────────────┘

    NOTE: The @torch.no_grad() decorator matches the official implementation.
    The indexer doesn't backpropagate gradients — in production GLM-5.2, it's
    pre-trained separately. For training from scratch, the random-but-causal
    token selection acts as attention regularization. The model learns to be
    robust to approximate attention through its main MLA weights.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank

        # The indexer has its OWN projections — completely separate from main attention!
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        # Learned per-head importance: "how much should each head's score be trusted?"
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, cos, sin, position_ids):
        """
        Select top-k most relevant tokens for each query position.

        Args:
            hidden_states: [B, S, hidden_size] — input to this layer
            q_resid: [B, S, q_lora_rank] — query residual from MLA's q_a_layernorm
            cos, sin: position embeddings
            position_ids: [B, S]

        Returns:
            topk_indices: [B, S, topk] — indices of selected tokens (int32)
        """
        B, S, _ = hidden_states.shape

        # --- Query: project from q_resid (shared with main attention's LoRA output) ---
        q = self.wq_b(q_resid).view(B, S, self.n_heads, self.head_dim)
        q_rot, q_pass = q.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # --- Key: project from hidden states (fresh, independent projection) ---
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, head_dim]
        k_rot, k_pass = k.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # --- Apply interleaved RoPE to both Q and K ---
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, n_heads, head_dim]
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, head_dim]

        # --- Multi-head relevance scoring ---
        # Each head independently scores every (query, key) pair
        # q: [B, S, n_heads, D] @ k^T: [B, 1, D, S] → [B, S, n_heads, S]
        scores = (
            torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        )
        scores = F.relu(scores)  # ReLU! Not softmax. This creates naturally sparse scores.

        # --- Weighted head aggregation ---
        # Learn which heads' opinions matter more, then combine
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
        weights = weights * (self.n_heads**-0.5)
        # [B, S, 1, n_heads] @ [B, S, n_heads, S] → [B, S, 1, S] → squeeze → [B, S, S]
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # --- Enforce causality: can't select future tokens! ---
        key_positions = torch.arange(S, device=hidden_states.device)
        causal = key_positions[None, None, :] > position_ids[:, :, None]
        index_scores = index_scores.masked_fill(causal, float("-inf"))

        # --- Select top-k most relevant tokens ---
        topk = min(self.index_topk, S)
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)


# ---------------------------------------------------------------------------
# 2h: Multi-Latent Attention (MLA) + DSA Integration
# ---------------------------------------------------------------------------


class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention (MLA) with DeepSeek Sparse Attention (DSA).

    MLA compresses queries and key-values through LoRA-style bottlenecks.
    This dramatically reduces KV-cache size during inference.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Query path:                                                           │
    │    x → q_a_proj (compress) → RMSNorm → q_b_proj (expand per-head)     │
    │    → split into [q_nope, q_rope] → apply RoPE to q_rope               │
    │                                                                        │
    │  KV path:                                                              │
    │    x → kv_a_proj (compress to [kv_latent + k_rope])                    │
    │    → kv_latent → RMSNorm → kv_b_proj (expand per-head)                │
    │    → split into [k_nope, value]                                        │
    │    → k_rope gets RoPE and is broadcast to all heads                    │
    │                                                                        │
    │  Then: q = [q_nope, q_rope], k = [k_nope, k_rope]                     │
    │  Standard scaled dot-product attention with DSA sparse masking          │
    └─────────────────────────────────────────────────────────────────────────┘

    Cross-layer DSA sharing:
    - "full" layers run the indexer to compute fresh top-k indices
    - "shared" layers reuse the previous full layer's indices (saves compute)
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim  # nope + rope
        self.v_head_dim = config.v_head_dim

        # === Query LoRA compression ===
        # hidden → compress → normalize → expand to per-head queries
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # === KV LoRA compression ===
        # hidden → compress to [kv_latent, k_rope_shared]
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # kv_latent → expand to per-head [k_nope, value]
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # === Output projection ===
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

        # === Attention scaling ===
        self.scaling = self.qk_head_dim ** (-0.5)

        # === DSA: indexer or shared ===
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else DSAIndexer(config, layer_idx)

    def forward(self, x, cos, sin, position_ids, prev_topk_indices=None):
        B, T, _ = x.shape

        # ============ Query Path ============
        # x → compress(768→384) → RMSNorm → expand(384→12*64=768)
        q_resid = self.q_a_layernorm(self.q_a_proj(x))  # [B, T, q_lora_rank=384]
        q = self.q_b_proj(q_resid)  # [B, T, num_heads * qk_head_dim]
        q = q.view(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2)  # [B, H, T, qk_head_dim]
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ============ KV Path ============
        # x → compress(768→160) → split [kv_latent(128), k_rope(32)]
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_compressed, k_rope = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # kv_latent → RMSNorm → expand(128→12*96=1152) → split [k_nope(32), v(64)]
        kv = self.kv_b_proj(self.kv_a_layernorm(k_compressed))
        kv = kv.view(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ============ Apply Interleaved RoPE ============
        # k_rope is shared across heads (MQA-style for the rope component)
        k_rope = k_rope.view(B, 1, T, self.qk_rope_head_dim)
        q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos, sin)
        k_rope = k_rope.expand(B, self.num_heads, T, -1)  # broadcast to all heads

        # Concatenate nope + rope components for final Q and K
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, H, T, qk_head_dim=64]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, H, T, qk_head_dim=64]

        # ============ DSA: Sparse Token Selection ============
        if self.indexer is not None:
            # "Full" layer: run indexer to get fresh top-k indices
            topk_indices = self.indexer(x, q_resid, cos, sin, position_ids)
        else:
            # "Shared" layer: reuse previous full layer's indices
            assert prev_topk_indices is not None, (
                f"Layer {self.layer_idx} is 'shared' DSA but got no previous top-k indices!"
            )
            topk_indices = prev_topk_indices

        # ============ Attention Computation ============
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [B, H, T, T]

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.full((T, T), torch.finfo(q.dtype).min, device=x.device, dtype=q.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask[None, None, :, :]

        # DSA sparse mask: ONLY attend to the indexer's top-k selected tokens
        # index_mask[b, t, t'] = True → position t' is NOT selected → mask it out
        index_mask = torch.ones(B, T, T, device=x.device, dtype=torch.bool)
        index_mask.scatter_(-1, topk_indices.long(), False)  # Set selected positions to False (unmasked)
        attn_weights = attn_weights.masked_fill(
            index_mask.unsqueeze(1),  # [B, 1, T, T] — broadcast across heads
            torch.finfo(q.dtype).min,
        )

        # Softmax + weighted sum of values
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, v_head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, topk_indices


# ---------------------------------------------------------------------------
# 2i: Transformer Decoder Layer
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """
    Pre-norm Transformer decoder layer.

    Structure:
        x → LayerNorm → MLA Attention → +residual → LayerNorm → MLP/MoE → +residual

    Layers 0..first_k_dense_replace use dense SwiGLU MLP.
    Remaining layers use Mixture-of-Experts (MoE).
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MultiLatentAttention(config, layer_idx)

        # Choose MLP type based on layer position
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoEBlock(config)
        else:
            self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, position_ids, prev_topk_indices=None):
        # Pre-norm → Attention → Residual
        residual = x
        x = self.input_layernorm(x)
        attn_out, topk_indices = self.self_attn(x, cos, sin, position_ids, prev_topk_indices)
        x = residual + attn_out

        # Pre-norm → MLP/MoE → Residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        return x, topk_indices


# ---------------------------------------------------------------------------
# 2j: Full Model (Base + CausalLM head)
# ---------------------------------------------------------------------------


class GLM5Model(nn.Module):
    """GLM-5.2 base model: token embeddings → N decoder layers → final RMSNorm."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.gradient_checkpointing = False

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.config.max_position_embeddings, (
            f"Sequence length {T} > max_position_embeddings {self.config.max_position_embeddings}"
        )

        x = self.embed_tokens(input_ids)

        # Compute position embeddings once (shared across all layers)
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(x, position_ids)

        # Forward through decoder layers
        # Each layer returns (hidden_states, topk_indices)
        # topk_indices propagate from "full" DSA layers to "shared" layers
        topk_indices = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # gradient_checkpointing.checkpoint does not support None inputs.
                # Pass a sentinel zero-tensor when topk_indices is None (layer 0 full-indexer layers),
                # and detect it inside with a flag. Simpler: just skip checkpointing for the very
                # first "full" layer (layer 0) which has no prev indices to receive.
                if topk_indices is None:
                    # Layer 0 is always a "full" DSA layer — run normally, then checkpoint the rest.
                    x, topk_indices = layer(x, cos, sin, position_ids, None)
                else:
                    x, topk_indices = torch.utils.checkpoint.checkpoint(
                        layer, x, cos, sin, position_ids, topk_indices,
                        use_reentrant=False,
                    )
            else:
                x, topk_indices = layer(x, cos, sin, position_ids, topk_indices)

        return self.norm(x)


class GLM5ForCausalLM(nn.Module):
    """
    GLM-5.2 for Causal Language Modeling.
    = GLM5Model (base) + Linear lm_head (vocab projection).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GLM5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GLM-5.2 conventions."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, MoEExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: [B, T] token indices
            targets: [B, T] target token indices (shifted by 1 in get_batch)

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar if targets provided, else None
        """
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        """
        Simple autoregressive generation with temperature + top-k sampling.

        No KV-cache for simplicity — recomputes the full context each step.
        This is slower but simpler, and matches Karpathy's nanoGPT style.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max context length if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_position_embeddings
                else idx[:, -self.config.max_position_embeddings :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def param_count(self):
        """Return a detailed parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        embed_params = self.model.embed_tokens.weight.numel()
        head_params = self.lm_head.weight.numel() if not self.config.tie_word_embeddings else 0
        non_embed = total - embed_params - head_params

        # MoE active params per token
        moe_layers = sum(1 for t in self.config.mlp_layer_types if t == "sparse")
        if moe_layers > 0:
            expert_params_per_layer = (
                2 * self.config.moe_intermediate_size * self.config.hidden_size
                + self.config.hidden_size * self.config.moe_intermediate_size
            )
            total_expert_params = expert_params_per_layer * self.config.n_routed_experts * moe_layers
            active_expert_params = expert_params_per_layer * self.config.num_experts_per_tok * moe_layers
            active_ratio = self.config.num_experts_per_tok / self.config.n_routed_experts
        else:
            total_expert_params = 0
            active_expert_params = 0
            active_ratio = 1.0

        active_params = total - total_expert_params + active_expert_params
        return {
            "total": total,
            "non_embedding": non_embed,
            "active_per_token": active_params,
            "moe_active_ratio": active_ratio,
        }


# =============================================================================
# Section 3: Data Loading
# =============================================================================
# Loads pre-tokenized binary data produced by dataprep_pretrain.py.
# Data format: train.bin / val.bin (uint16 memmap) + meta.json.
#
# Run dataprep_pretrain.py first to prepare the data:
#   python dataprep_pretrain.py              # Full 3.3B tokens
#   python dataprep_pretrain.py --total_tokens 10000000  # Quick 10M test
# =============================================================================


def load_pretrain_data(data_dir):
    """
    Load pre-tokenized binary data from data_dir.

    Expects:
      data_dir/train.bin  -- binary token file (uint16 or uint32)
      data_dir/val.bin    -- binary token file (uint16 or uint32)
      data_dir/meta.json  -- metadata (vocab_size, dtype, token counts)

    Returns:
      train_data: np.memmap of training tokens
      val_data: np.memmap of validation tokens

    Raises:
      FileNotFoundError if data files are missing.
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    meta_path = os.path.join(data_dir, "meta.json")

    # --- Validate files exist ---
    for path, name in [(train_path, "train.bin"), (val_path, "val.bin"), (meta_path, "meta.json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"  [ERR] {name} not found at: {path}\n"
                f"        Run dataprep_pretrain.py first to prepare the data:\n"
                f"          python dataprep_pretrain.py\n"
                f"        Or for a quick test:\n"
                f"          python dataprep_pretrain.py --total_tokens 10000000"
            )

    # --- Load metadata ---
    with open(meta_path, "r") as f:
        meta = json.load(f)

    dtype_str = meta.get("dtype", "uint16")
    dtype = np.uint16 if dtype_str == "uint16" else np.uint32
    train_tokens = meta.get("train_tokens", 0)
    val_tokens = meta.get("val_tokens", 0)

    print(f"  [Data]")
    print(f"    Tokenizer:   {meta.get('tokenizer', 'unknown')}")
    print(f"    Vocab size:  {meta.get('vocab_size', 'unknown')}")
    print(f"    Dtype:       {dtype_str}")
    print(f"    Train:       {train_tokens:,} tokens")
    print(f"    Val:         {val_tokens:,} tokens")
    print(f"    Total:       {train_tokens + val_tokens:,} tokens")
    print(f"    Sources:     {', '.join(meta.get('sources', []))}")

    # --- Memory-map the binary files ---
    # memmap reads directly from disk without loading into RAM.
    # This is critical for 3.3B tokens (~7GB) on a 6GB VRAM machine.
    train_data = np.memmap(train_path, dtype=dtype, mode="r")
    val_data = np.memmap(val_path, dtype=dtype, mode="r")

    return train_data, val_data


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    """
    Sample a random batch of token sequences from memmap data.

    Uses .copy() on numpy slices to avoid torch tensor issues with
    non-writable memmap arrays.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64).copy()) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64).copy()) for i in ix])
    return x.to(device), y.to(device)


# =============================================================================
# Section 4: Training
# =============================================================================
# AdamW optimizer with cosine LR schedule, gradient accumulation,
# mixed precision, gradient checkpointing, periodic eval + checkpointing.
# =============================================================================


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr, stable_iters=0):
    """WSD (Warmup-Stable-Decay) learning rate schedule.

    Phases:
      1. Warmup:  steps [0, warmup_iters)           — linear ramp 0 → learning_rate
      2. Stable:  steps [warmup_iters, stable_iters) — constant at learning_rate
      3. Decay:   steps [stable_iters, lr_decay_iters] — cosine decay → min_lr

    If stable_iters <= warmup_iters (default: 0), this reduces to the standard
    cosine schedule with warmup (backward-compatible).
    """
    # Phase 1: Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # Phase 3 ended: hold at min_lr
    if it > lr_decay_iters:
        return min_lr
    # Phase 2: Stable (constant LR) — only if stable_iters is set
    if stable_iters > warmup_iters and it < stable_iters:
        return learning_rate
    # Phase 3: Cosine decay from learning_rate → min_lr
    decay_start = max(stable_iters, warmup_iters)
    decay_ratio = (it - decay_start) / (lr_decay_iters - decay_start)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, block_size, batch_size, device, ctx):
    """Estimate loss on train and val splits (averaged over eval_iters batches)."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


def train(args):
    """Main training function."""
    # --- Device Setup ---
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  >> Let's Reproduce GLM-5.2 (GLM MoE DSA) -- From Scratch!")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"  GPU:    {torch.cuda.get_device_name()}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  [OK] TF32 Tensor Cores enabled")

    # --- Data ---
    train_data, val_data = load_pretrain_data(args.data_dir)

    # --- Model ---
    config = GLM5Config()
    model = GLM5ForCausalLM(config)
    counts = model.param_count()

    print(f"\n  [Model Architecture]")
    print(f"    Hidden size:        {config.hidden_size}")
    print(f"    Layers:             {config.num_hidden_layers} "
          f"({config.first_k_dense_replace} dense + "
          f"{config.num_hidden_layers - config.first_k_dense_replace} MoE)")
    print(f"    Attention heads:    {config.num_attention_heads}")
    print(f"    Q LoRA rank:        {config.q_lora_rank}  ->  qk_head_dim: {config.qk_head_dim} "
          f"(nope:{config.qk_nope_head_dim} + rope:{config.qk_rope_head_dim})")
    print(f"    KV LoRA rank:       {config.kv_lora_rank}  ->  v_head_dim: {config.v_head_dim}")
    print(f"    Experts:            {config.n_routed_experts} routed "
          f"(top-{config.num_experts_per_tok}) + {config.n_shared_experts} shared")
    print(f"    DSA index_topk:     {config.index_topk}")
    print(f"    Indexer pattern:    {''.join('F' if t == 'full' else 'S' for t in config.indexer_types)}")
    print(f"\n  [Parameters]")
    print(f"    Total:              {counts['total']:>12,}")
    print(f"    Non-embedding:      {counts['non_embedding']:>12,}")
    print(f"    Active per token:   {counts['active_per_token']:>12,}  "
          f"({counts['moe_active_ratio']:.0%} of experts active)")
    print(f"    VRAM (est. train):  ~{counts['total'] * 16 / 1e9:.1f} GB  "
          f"(weights + optimizer + gradients)")

    model = model.to(device)

    # --- Gradient Checkpointing ---
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing = True
        print(f"\n  [OK] Gradient checkpointing: ON (saves ~40% VRAM, ~30% slower)")

    # --- Mixed Precision ---
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        print(f"  [OK] Mixed precision: bfloat16")
    elif device == "cuda":
        dtype = torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        print(f"  [OK] Mixed precision: float16")
    else:
        dtype = torch.float32
        ctx = torch.amp.autocast(device_type="cpu", enabled=False)
        print(f"  [WARN] No mixed precision (CPU mode)")

    # GradScaler only needed for float16 (bfloat16 doesn't need scaling)
    # On CPU, GradScaler must be disabled entirely (no CUDA streams available)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device == "cuda"))

    # --- Dry Run: verify forward pass and check VRAM ---
    print(f"\n  Verifying forward pass...")
    try:
        with torch.no_grad():
            dummy = torch.randint(0, config.vocab_size, (1, args.block_size), device=device)
            with ctx:
                _, test_loss = model(dummy, dummy)
            print(f"  [OK] Forward pass OK (dummy loss={test_loss.item():.4f})")
            if device == "cuda":
                print(f"  [OK] VRAM after forward: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
    except torch.cuda.OutOfMemoryError:
        print(f"  [ERR] OOM during forward pass! Try reducing --batch_size or --block_size")
        return

    # --- torch.compile ---
    raw_model = model  # Keep a reference to the un-compiled model for saving
    if args.compile and device == "cuda":
        print(f"  [OK] Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    # --- Optimizer ---
    # Separate weight decay: only for 2D+ params (weight matrices), not biases/norms
    decay_params = []
    no_decay_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        fused=(device == "cuda"),
    )

    tokens_per_step = args.batch_size * args.block_size * args.gradient_accumulation_steps
    print(f"\n  [Training Configuration]")
    print(f"    Batch size:         {args.batch_size} x {args.gradient_accumulation_steps} "
          f"grad accum = {args.batch_size * args.gradient_accumulation_steps} effective")
    print(f"    Sequence length:    {args.block_size}")
    print(f"    Tokens per step:    {tokens_per_step:,}")
    print(f"    Max iterations:     {args.max_iters:,}")
    print(f"    Total tokens:       ~{tokens_per_step * args.max_iters:,}")
    print(f"    Learning rate:      {args.learning_rate} -> {args.min_lr} (cosine)")
    print(f"    Warmup:             {args.warmup_iters} steps")
    print(f"{'='*70}\n")

    # --- Resume from Checkpoint (if requested or if ckpt.pt exists) ---
    start_iter = 0
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    if args.resume and os.path.exists(ckpt_path):
        print(f"  [RESUME] Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter_num", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  [RESUME] Resuming from step {start_iter} (best val loss: {best_val_loss:.4f})")

    # --- Training Loop ---
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    tokens_processed = 0

    for iter_num in range(start_iter, args.max_iters):
        # Update learning rate (WSD schedule: warmup → stable → cosine decay)
        lr = get_lr(iter_num, args.warmup_iters, args.lr_decay_iters, args.learning_rate, args.min_lr, args.stable_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Periodic Evaluation ---
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                args.eval_iters, args.block_size, args.batch_size, device, ctx,
            )
            print(
                f"  step {iter_num:>5d} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {lr:.2e}"
            )

            # Save latest checkpoint at every eval interval so progress is never lost
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(ckpt, ckpt_path)
            print(f"           [SAVED] latest checkpoint to {ckpt_path} (step {iter_num})")

            # Save separate best checkpoint when val loss improves
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt["best_val_loss"] = best_val_loss
                best_ckpt_path = os.path.join(args.out_dir, "ckpt_best.pt")
                torch.save(ckpt, best_ckpt_path)
                print(f"           [SAVED] BEST checkpoint to {best_ckpt_path} (val_loss={best_val_loss:.4f})")

        # --- Gradient Accumulation Loop ---
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.gradient_accumulation_steps):
            X, Y = get_batch("train", train_data, val_data, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            tokens_processed += X.numel()

        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # --- Logging ---
        if iter_num > 0 and iter_num % args.log_interval == 0:
            dt = time.time() - t0
            tps = tokens_processed / dt if dt > 0 else 0
            lossf = loss.item() * args.gradient_accumulation_steps
            vram = ""
            if device == "cuda":
                vram = f" | VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            print(f"  step {iter_num:>5d} | loss {lossf:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s{vram}")
            t0 = time.time()
            tokens_processed = 0

    print(f"\n{'='*70}")
    print(f"  [DONE] Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {os.path.join(args.out_dir, 'ckpt.pt')}")
    print(f"{'='*70}")


# =============================================================================
# Section 5: Text Generation / Sampling
# =============================================================================


def sample(args):
    """Generate text from a trained checkpoint."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt_path = args.ckpt or os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [ERR] Checkpoint not found at {ckpt_path}")
        print(f"    Train first with: python train_glm5.py")
        return

    print(f"  Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = GLM5ForCausalLM(config)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    # Encode prompt
    enc = tiktoken.get_encoding("gpt2")
    prompt = args.prompt or "\n"
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"\n  Prompt: {prompt!r}")
    print(f"  {'-'*60}")

    # Generate — use bfloat16 on CUDA, float32 on CPU (autocast doesn't support bf16 on CPU)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        gen_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif device == "cuda":
        gen_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        import contextlib
        gen_ctx = contextlib.nullcontext()
    with gen_ctx:
        output = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    generated = enc.decode(output[0].tolist())
    print(generated)
    print(f"  {'-'*60}\n")


# =============================================================================
# Section 6: Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Let's Reproduce GLM-5.2 (GLM MoE DSA) -- From Scratch!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Mode ---
    parser.add_argument("--eval_only", action="store_true", help="Generate text only (no training)")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing out_dir/ckpt.pt")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for generation")

    # --- Training ---
    parser.add_argument("--out_dir", type=str, default="out_glm5", help="Output directory for checkpoints")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory containing train.bin, val.bin, meta.json")
    parser.add_argument("--max_iters", type=int, default=800000, help="Training iterations (800K for 3.3B tokens)")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size per step")
    parser.add_argument("--block_size", type=int, default=512, help="Context/sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate (end of cosine)")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="LR warmup iterations")
    parser.add_argument("--lr_decay_iters", type=int, default=800000, help="Cosine decay length (match max_iters)")
    parser.add_argument("--stable_iters", type=int, default=0, help="WSD: keep LR at peak until this step, then cosine decay (0=standard cosine)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (0=disable)")

    # --- Efficiency ---
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True,
        help="Enable gradient checkpointing (default: on, saves VRAM)",
    )
    parser.add_argument(
        "--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
        help="Disable gradient checkpointing",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (faster, needs warmup)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")

    # --- Evaluation ---
    parser.add_argument("--eval_interval", type=int, default=2000, help="Evaluate every N steps")
    parser.add_argument("--eval_iters", type=int, default=50, help="Batches per evaluation")
    parser.add_argument("--log_interval", type=int, default=100, help="Log loss every N steps")

    # --- Generation ---
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k sampling")

    args = parser.parse_args()

    if args.eval_only:
        sample(args)
    else:
        train(args)
        # Generate a sample after training completes
        print("\n  >> Generating sample text from the trained model...\n")
        args.prompt = args.prompt or "First Citizen:\n"
        sample(args)


if __name__ == "__main__":
    main()
