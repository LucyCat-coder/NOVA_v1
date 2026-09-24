import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} должен делиться на num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        q, k, v = self.qkv(x).chunk(
            3,
            dim=-1,
        )

        q = (
            q.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        k = (
            k.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        v = (
            v.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout_p
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        out = (
            out
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                channels,
            )
        )

        return self.out_proj(out)