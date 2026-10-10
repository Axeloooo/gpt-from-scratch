"""Lectures 7 and 9: a decoder-only transformer (GPT-2 architecture) and its optimizer setup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Defaults are GPT-2 124M. `vocab_size` 50304 is 50257 rounded up to a multiple of 128."""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, flash: bool = True) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # query, key, value at once
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[assignment]
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = flash
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # the explicit version from lecture 7: scores, causal mask, softmax, weighted sum
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))  # type: ignore[index]
            y = F.softmax(att, dim=-1) @ v
        out: torch.Tensor = self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return out


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.c_proj(self.gelu(self.c_fc(x)))
        return out


class Block(nn.Module):
    """Pre-norm residual block: communication (attention) then computation (MLP)."""

    def __init__(self, config: GPTConfig, flash: bool = True) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, flash)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        out: torch.Tensor = x + self.mlp(self.ln_2(x))
        return out


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, flash: bool = True) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config, flash) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight  # weight tying, as in GPT-2
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            # residual paths add 2 contributions per layer; keep the stream's variance in check
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block size {self.config.block_size}")
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        for block in cast(nn.ModuleList, self.transformer["h"]):
            x = block(x)
        logits = self.lm_head(self.transformer["ln_f"](x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on matrices (weights, embeddings) but not on biases/LayerNorm."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=1e-8, fused=fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size :])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, nxt), dim=1)
        self.train(was_training)
        return idx
