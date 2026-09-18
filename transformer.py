"""From-scratch Transformer encoder with Rotary Positional Embeddings (RoPE)
for sequence classification over mel-spectrogram frames.
"""

import math

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary positional embedding applied to attention queries/keys."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = torch.exp((torch.arange(0, head_dim, 2).float() * math.log(10_000)) / head_dim)
        T = torch.arange(max_seq_len).float()
        freq = torch.einsum("i,j->ij", T, inv_freq)

        self.register_buffer("cos", freq.cos())
        self.register_buffer("sin", freq.sin())

    def forward(self, x):
        T = x.shape[1]

        x1 = x[..., ::2]   # (batch, seq_len, num_head, head_dim/2)
        x2 = x[..., 1::2]  # (batch, seq_len, num_head, head_dim/2)

        cos = self.cos[:T][None, :, None, :]
        sin = self.sin[:T][None, :, None, :]

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        output = torch.stack([x1_rot, x2_rot], dim=-1)
        return output.flatten(-2)


class ModelAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K."""

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_head == 0, f"{config.embed_dim} % {config.num_head} != 0"
        self.num_head = config.num_head
        self.head_dim = config.embed_dim // config.num_head
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.rope = RoPE(head_dim=self.head_dim)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.rope(q), self.rope(k)

        # q = k = v = (B, num_head, T, head_dim)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # (B, num_head, T, head_dim)
        return out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, embed_dim)


class AudioTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_attn = ModelAttention(config)
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )

    def forward(self, x):
        x = x + self.ln_1(self.model_attn(x))
        x = x + self.ln_2(self.mlp(x))
        return x


class AudioTransformer(nn.Module):
    def __init__(self, config, name: str = "audio_transformer"):
        super().__init__()
        self._name = name
        self.norm = nn.LayerNorm(config.embed_dim)
        self.input_proj = nn.Linear(config.n_mels, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.num_class)
        self.transformer_block = nn.ModuleList(
            AudioTransformerBlock(config) for _ in range(config.num_transformer_layers)
        )

    @property
    def name(self):
        return self._name

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.transformer_block:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.out_proj(x)
