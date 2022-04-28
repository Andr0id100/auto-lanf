from torch import nn
from common import FeedForward, PositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512, ff_dim=2048):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            hidden_dim, num_heads)
        self.attention_normalize = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.ff_normalize = nn.LayerNorm(hidden_dim)

    def forward(self, x, src_mask, src_padding_mask):
        x = self.attention_normalize(x + self.multi_head_attention(
            x, x, x, key_padding_mask=src_padding_mask, attn_mask=src_mask)[0])
        x = self.ff_normalize(x + self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(self,
                 block_count=6,
                 head_count=8,
                 hidden_dim=512,
                 ff_dim=2048,
                 vocab_size=50000,
                 max_length=1024):
        super().__init__()
        self.embedding = nn.Embedding(
            embedding_dim=hidden_dim,
            num_embeddings=vocab_size,
        )
        self.positional_encoding = PositionalEncoding(
            max_length=max_length, hidden_dim=hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(hidden_dim=hidden_dim, ff_dim=ff_dim) for _ in range(block_count)])

    def forward(self, x, src_mask, src_padding_mask):
        x = self.embedding(x)
        x += self.positional_encoding(x)

        for block in self.encoder_blocks:
            x = block(x, src_mask, src_padding_mask)

        return x
