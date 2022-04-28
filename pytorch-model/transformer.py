from torch import nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 encoder_block_count=6,
                 decoder_block_count=6,
                 hidden_dim=512,
                 ff_dim=2048,
                 vocab_size=50000,
                 max_length=1024,
                 head_count=8):
        super().__init__()
        self.encoder = Encoder(block_count=encoder_block_count, hidden_dim=hidden_dim,
                               ff_dim=ff_dim, vocab_size=vocab_size, max_length=max_length, head_count=head_count)
        self.decoder = Decoder(block_count=decoder_block_count, hidden_dim=hidden_dim,
                               ff_dim=ff_dim, vocab_size=vocab_size, max_length=max_length, head_count=head_count)

        self.vocab_classifier = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source, target, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        encoded = self.encoder(source, src_mask, src_padding_mask)
        decoded = self.decoder(target, encoded, tgt_mask, tgt_padding_mask)
        classified = self.softmax(self.vocab_classifier(decoded))
        return classified
