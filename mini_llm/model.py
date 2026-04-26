"""
Minimal Transformer Language Model.

Architecture:
- Token + Positional embeddings
- 2 transformer decoder layers
- 2 attention heads
- Embedding dim: 32
- Context window: 16 tokens
- ~18K parameters total

Every component is kept minimal but functionally complete,
so each piece maps directly to a real LLM concept.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal mask.

    This is the core mechanism that lets the model decide
    which previous tokens are relevant for predicting the next one.
    We store attention weights for visualization.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Store last attention weights for visualization
        self.last_attn_weights = None

    def forward(self, x, mask=None):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.W_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.W_v(x).view(B, T, self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask: prevent attending to future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn_weights.detach()

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Simple feed-forward network (the 'thinking' part of each layer)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer decoder block: attention + feed-forward + layer norms."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-norm architecture (like modern LLMs)
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """
    The complete mini language model.

    Hyperparameters chosen to be the smallest that still
    demonstrate real transformer behavior:
    - d_model=32:  small but enough for 2-head attention
    - n_heads=2:   minimum to show multi-head concept
    - n_layers=2:  minimum to show depth
    - d_ff=64:     2x expansion (real LLMs use 4x)
    - max_seq=16:  tiny context window
    """

    def __init__(self, vocab_size: int, d_model=32, n_heads=2,
                 n_layers=2, d_ff=64, max_seq_len=16):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (like real LLMs)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max {self.max_seq_len}"
        )

        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        mask = torch.tril(torch.ones(T, T, device=idx.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_final(x)
        logits = self.head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_attention_maps(self) -> list:
        maps = []
        for layer in self.layers:
            if layer.attn.last_attn_weights is not None:
                maps.append(layer.attn.last_attn_weights)
        return maps

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=10, temperature=1.0,
                 top_k=None, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            if temperature < 0.01:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

        return idx
