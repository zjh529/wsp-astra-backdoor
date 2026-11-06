"""SmallTransformer matching arXiv:2610.07334 target-model dims.

1-layer pre-LN transformer classifier:
  d_model=8, n_heads=2 (head_dim=4), mlp_dim=16, seq_len=4, vocab=1000, n_classes=10.
Readout: classifier head on the residual stream at the LAST position.

Parameter names (state_dict keys) are explicit so weight-only auditors can find them:
  tok_emb.weight [1000,8], pos_emb [4,8],
  ln1.{weight,bias}, attn.W_q/W_k/W_v/W_o.{weight,bias} [8,8],
  ln2.{weight,bias}, mlp.fc1 [16,8], mlp.fc2 [8,16],
  ln_f.{weight,bias}, head [10,8].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = dict(d_model=8, n_heads=2, mlp_dim=16, seq_len=4, vocab=1000, n_classes=10)


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


class SmallTransformer(nn.Module):
    def __init__(self, d_model=8, n_heads=2, mlp_dim=16, seq_len=4, vocab=1000, n_classes=10):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.1)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_dim)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens):
        x = self.tok_emb(tokens) + self.pos_emb[: tokens.shape[1]]
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return self.head(self.ln_f(x[:, -1]))


def n_params(m: nn.Module, exclude_embedding: bool = False) -> int:
    return sum(p.numel() for n, p in m.named_parameters() if not (exclude_embedding and n.startswith("tok_emb")))
