from torch import nn
from common import FeedForward, PositionalEncoding


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, ff_dim=2048):
        super().__init__()
        self.masked_decoder_attention = nn.MultiheadAttention(
            hidden_dim, num_heads)
        self.masked_normalize = nn.LayerNorm(hidden_dim)
        self.encoder_decoder_attention = nn.MultiheadAttention(
            hidden_dim, num_heads)
        self.encoder_decoder_normalize = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.ff_normalize = nn.LayerNorm(hidden_dim)

    def forward(self, x, encoded, tgt_mask, tgt_padding_mask):
        x = self.masked_normalize(x + self.masked_decoder_attention(
            x, x, x, key_padding_mask=tgt_padding_mask, attn_mask=tgt_mask)[0])

        x = self.encoder_decoder_normalize(
            x + self.encoder_decoder_attention(x, encoded, encoded, )[0])
        x = self.ff_normalize(x + self.feed_forward(x))
        return x


class Decoder(nn.Module):
    def __init__(self,
                 head_count=8,
                 block_count=6,
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

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(hidden_dim=hidden_dim, ff_dim=ff_dim) for _ in range(block_count)])

    def forward(self, x, encoded, tgt_mask, tgt_padding_mask):
        x = self.embedding(x)
        x += self.positional_encoding(x)

        for block in self.decoder_blocks:
            x = block(x, encoded, tgt_mask, tgt_padding_mask)
        return x
