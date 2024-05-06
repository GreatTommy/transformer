import torch
from torch import nn

NUM_LAYERS = 2
DIM_EMBED = 32
NUM_HEADS = 2
DIM_HEAD = DIM_EMBED // NUM_HEADS
DIM_FF = 32
SEQ_LEN = 12
VOCAB_SIZE = len(range(32, 127))  # ASCII range

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lookup = nn.Embedding(VOCAB_SIZE, DIM_EMBED)

    def forward(self, x):
        return self.lookup(x) * (DIM_EMBED**0.5)


class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        positions = torch.arange(0, SEQ_LEN).unsqueeze(1)
        frequencies = torch.pow(10000, torch.arange(0, DIM_EMBED, 2) / DIM_EMBED)
        pe = torch.zeros(SEQ_LEN, DIM_EMBED)  # word position x embedding dimension
        pe[:, 0::2] = torch.sin(positions / frequencies)
        pe[:, 1::2] = torch.cos(positions / frequencies)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(DIM_EMBED, DIM_HEAD)
        self.k = nn.Linear(DIM_EMBED, DIM_HEAD)
        self.v = nn.Linear(DIM_EMBED, DIM_HEAD)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        q = self.q(x)  # (B, S, D) (batch size, sequence length, embedding dimension)
        k = self.k(x)  # (B, S, D)
        v = self.v(x)  # (B, S, D)
        weights = torch.matmul(q, k.transpose(-2, -1)) / (DIM_HEAD**0.5)  # (B, S, S)
        weights = weights.masked_fill(self.mask, float("-inf"))
        weights = torch.softmax(weights, dim=-1)  # (B, S, S)
        return torch.matmul(weights, v)  # (B, S, D)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(NUM_HEADS)])
        self.fc = nn.Linear(NUM_HEADS * DIM_HEAD, DIM_EMBED)

    def forward(self, x):
        return self.fc(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM_EMBED, DIM_FF), nn.ReLU(), nn.Linear(DIM_FF, DIM_EMBED)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM_EMBED)
        self.norm2 = nn.LayerNorm(DIM_EMBED)
        self.attn = MultiHeadAttention()
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = Embedder()
        self.pos_enc = PositionalEncoder()
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(DIM_EMBED)
        self.fc = nn.Linear(DIM_EMBED, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedder(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.fc(x)