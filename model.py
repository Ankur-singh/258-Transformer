import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, **kwargs)

    def forward(self, x):
        B, T, C = x.shape
        mask = MaskedMultiHeadAttention.create_mask(T).to(x.device)
        return self.mha(x, x, x, attn_mask=mask)

    # https://discuss.pytorch.org/t/the-way-to-implement-attention-mask-uni-direction-attention-in-transformerdecoder/73124/4
    @staticmethod
    def create_mask(size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.2):
        super().__init__()
        self.mmha = MaskedMultiHeadAttention(emb_dim, num_heads)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = FeedForward(emb_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mmha(x)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_blocks, num_heads, max_context, dropout=0.2
    ):
        super().__init__()
        self.max_context = max_context
        self.tkn_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(self.max_context, emb_dim)
        self.blocks = nn.Sequential(
            *[Block(emb_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.lmh = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        tkn_emb = self.tkn_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = pos_emb + tkn_emb
        x = self.blocks(x)
        logits = self.lmh(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_tokens):
        for _ in range(n_tokens):
            idx_crop = idx[:, -self.max_context :]
            logits, _ = self(idx_crop)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
