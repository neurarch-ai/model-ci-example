"""DocLM: a 12-layer, 768-wide decoder that completes internal documents.

GPT-2-small shaped (about 135M parameters with a 32K vocabulary). The width splits
evenly across 12 heads (head_dim 64) and the model returns raw logits, so
``nn.CrossEntropyLoss`` applies log-softmax itself.

The context length is the one number here that the training job also reads: the
batch and the sequence length it trains at live in ``configs/pretrain.yaml``, and
``.neurarch.yml`` points the plan at that file so the bot sizes the job it will
actually run.
"""

import torch
import torch.nn as nn

VOCAB_SIZE = 32000
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
CONTEXT = 2048


class Block(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=12, dropout=dropout, batch_first=True
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
        return x + self.mlp(self.ln2(x))


class DocLM(nn.Module):
    def __init__(self, context: int = CONTEXT, dropout: float = 0.0):
        super().__init__()
        self.context = context
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(context, EMBED_DIM)
        self.blocks = nn.ModuleList([Block(dropout) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        assert t <= self.context, "sequence longer than the context"
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.head(self.ln_f(x))


if __name__ == "__main__":
    model = DocLM()
    idx = torch.randint(0, VOCAB_SIZE, (2, 64))
    print(model(idx).shape, sum(p.numel() for p in model.parameters()))
