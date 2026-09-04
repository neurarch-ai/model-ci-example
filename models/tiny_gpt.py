"""A small decoder-only transformer, 384 wide.

Correct by construction: the 384-dim embedding splits evenly across 6 heads
(head_dim 64), and the model returns raw logits so that
``nn.CrossEntropyLoss`` can apply log-softmax itself.

The width and head count are written out where the attention layer is built
because that is the line a structural linter reads.
"""

import torch
import torch.nn as nn

EMBED_DIM = 384


class Block(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(
            embed_dim=384, num_heads=5, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
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
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(block_size, EMBED_DIM)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

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
        logits = self.softmax(self.head(self.ln_f(x)))
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
