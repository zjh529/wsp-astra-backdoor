"""SmallTransformer matching arXiv:2610.07334 target-model dims (default),
with optional wider / deeper variants for Phase P2 scale axis.

Default (Phase T / P1):
  d_model=8, n_heads=2 (head_dim=4), mlp_dim=16, seq_len=4, vocab=1000, n_classes=10, n_layers=1.

Parameter names (state_dict keys) for the 1-layer case stay explicit so weight-only auditors can find them:
  tok_emb.weight [V,d], pos_emb [T,d],
  ln1.{weight,bias}, attn.W_q/W_k/W_v/W_o.{weight,bias} [d,d],
  ln2.{weight,bias}, mlp.fc1 [m,d], mlp.fc2 [d,m],
  ln_f.{weight,bias}, head [C,d].

When n_layers>1 the body is `blocks.{i}.{ln1,attn,ln2,mlp}` (ModuleList); embeddings / ln_f / head unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = dict(d_model=8, n_heads=2, mlp_dim=16, seq_len=4, vocab=1000, n_classes=10, n_layers=1)


class Attention(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h, self.dh = h, d // h
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape

        def split(t):
            return t.view(B, T, self.h, self.dh).transpose(1, 2)

        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.softmax(-1)  # bidirectional (classifier reads last position)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.W_o(y)


class MLP(nn.Module):
    def __init__(self, d: int, m: int):
        super().__init__()
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d: int, h: int, m: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d, m)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(
        self,
        d_model=8,
        n_heads=2,
        mlp_dim=16,
        seq_len=4,
        vocab=1000,
        n_classes=10,
        n_layers=1,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.1)
        if self.n_layers == 1:
            # Flat layout identical to Phase T / P1 (key-compatible)
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = Attention(d_model, n_heads)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = MLP(d_model, mlp_dim)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, mlp_dim) for _ in range(self.n_layers)]
            )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens):
        x = self.tok_emb(tokens) + self.pos_emb[: tokens.shape[1]]
        if self.n_layers == 1:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            for blk in self.blocks:
                x = blk(x)
        return self.head(self.ln_f(x[:, -1]))


def make_arch(d_model=8, n_heads=2, n_layers=1, seq_len=4, vocab=1000, n_classes=10):
    """Build an arch dict with mlp_dim = 2 × d_model (Phase T / P2 recipe)."""
    return dict(
        d_model=d_model,
        n_heads=n_heads,
        mlp_dim=2 * d_model,
        seq_len=seq_len,
        vocab=vocab,
        n_classes=n_classes,
        n_layers=n_layers,
    )


def n_params(m: nn.Module, exclude_embedding: bool = False) -> int:
    return sum(
        p.numel()
        for n, p in m.named_parameters()
        if not (exclude_embedding and n.startswith("tok_emb"))
    )
