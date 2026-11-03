"""
Lightweight 6-layer Transformer decoder for text-to-molecule translation.

Implements the architecture described in Section 3.3 of the article:
  - 6-layer Transformer decoder (PoC: 4-layer for faster CPU training)
  - d_model = 256, 4 attention heads, FFN expansion 4x (PoC: d_model=128)
  - ~12.4M parameters at production config
  - Causal self-attention with KV-cache for fast autoregressive generation
  - INT8 dynamic quantization hook for inference acceleration
  - Tied input/output embeddings (parameter efficiency)

Author: Methodology component #3 - lightweight decoder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
    """Configuration for the lightweight Transformer decoder."""
    vocab_size: int = 1024         # set from tokenizer at runtime
    d_model: int = 128             # Production: 256. PoC: 128 (CPU-friendly)
    n_heads: int = 4
    n_layers: int = 4              # Production: 6. PoC: 4 (faster training)
    ffn_mult: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    # Special token IDs (set from tokenizer at runtime)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    nl_sep_id: int = 4

    @property
    def num_parameters_millions(self) -> float:
        """Estimate parameter count in millions (for sanity check)."""
        emb = self.vocab_size * self.d_model
        pos = self.max_seq_len * self.d_model
        layer = 4 * self.d_model ** 2 + 2 * self.d_model * (self.ffn_mult * self.d_model)
        total = emb + pos + self.n_layers * layer
        return total / 1e6


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional KV-cache."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        self._cache_k = None
        self._cache_v = None

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if use_cache:
            if self._cache_k is not None:
                k = torch.cat([self._cache_k, k], dim=2)
                v = torch.cat([self._cache_v, v], dim=2)
            self._cache_k = k
            self._cache_v = v

        Tq = q.size(2)
        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(Tq, Tk, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        x = x + self.ffn(self.ln2(x))
        return x


class LightweightDecoder(nn.Module):
    """6-layer (PoC: 4-layer) Transformer decoder with tied embeddings."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.85,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation with nucleus sampling + KV-cache."""
        self.eval()
        device = prompt_ids.device
        for block in self.blocks:
            block.attn.reset_cache()

        out = prompt_ids
        logits = self.forward(out, use_cache=use_cache)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)

        for _ in range(max_new_tokens):
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum - sorted_probs) > top_p
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            next_tok = sorted_idx.gather(
                -1, torch.multinomial(sorted_probs, num_samples=1)
            )

            out = torch.cat([out, next_tok], dim=1)
            if next_tok.item() == self.cfg.eos_id:
                break

            if use_cache:
                logits = self.forward(next_tok, use_cache=True)
            else:
                logits = self.forward(out, use_cache=False)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

        return out


def count_parameters(model: nn.Module) -> Tuple[int, float]:
    """Return (total_params, params_in_millions)."""
    total = sum(p.numel() for p in model.parameters())
    return total, total / 1e6


def quantize_int8(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to all Linear layers."""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )