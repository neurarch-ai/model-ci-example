"""A small decoder-only transformer.

Correct by construction: embed_dim 384 splits evenly across 6 heads
(head_dim 64), and the model returns raw logits so that
``nn.CrossEntropyLoss`` can apply log-softmax itself.
"""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        block_size: int = 256,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        assert t <= self.block_size, "sequence longer than block_size"
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        logits = self.head(self.ln_f(x))
        return logits


def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token loss on raw logits. CrossEntropyLoss applies log-softmax internally."""
    criterion = nn.CrossEntropyLoss()
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


if __name__ == "__main__":
    model = TinyGPT()
    idx = torch.randint(0, 8192, (2, 32))
    out = model(idx)
    print(out.shape, loss(out, idx).item())
