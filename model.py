import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
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


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, mask=True, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.mask = mask
        if self.mask:
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head, max_context):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.mha = MultiHeadAttention(n_embd, n_head, block_size=max_context)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
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
            *[Block(emb_dim, num_heads, max_context) for _ in range(num_blocks)]
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
    def generate(self, idx, n_tokens, temperature=1.0, top_k=None):
        for _ in range(n_tokens):
            idx_crop = idx[:, -self.max_context :]
            logits, _ = self(idx_crop)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C)
            logits /= temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
