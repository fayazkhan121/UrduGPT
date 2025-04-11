import torch.nn as nn
from urdugpt_step3_embeddings import EmbeddingLayer, PositionalEncoding
from urdugpt_step6_encoder import Encoder
from urdugpt_step7_decoder import Decoder

# 🧠 Final linear layer to turn decoder output into logits over vocab
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)  # shape: (B, seq_len, vocab_size)

# 🏗️ Transformer = encoder + decoder + embeddings + projection
class Transformer(nn.Module):
    def __init__(self,
                 source_vocab_size, target_vocab_size,
                 source_seq_len, target_seq_len,
                 d_model=512, num_layers=6, num_heads=8,
                 d_ff=2048, dropout_rate=0.1):
        super().__init__()

        # Encoder parts
        self.source_embed = EmbeddingLayer(d_model, source_vocab_size)
        self.source_pos = PositionalEncoding(d_model, source_seq_len, dropout_rate)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout_rate)

        # Decoder parts
        self.target_embed = EmbeddingLayer(d_model, target_vocab_size)
        self.target_pos = PositionalEncoding(d_model, target_seq_len, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout_rate)

        # Final projection
        self.projection = ProjectionLayer(d_model, target_vocab_size)

        # Initialize weights (like in "Attention Is All You Need")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def encode(self, src, src_mask):
        x = self.source_embed(src)
        x = self.source_pos(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.target_embed(tgt)
        x = self.target_pos(x)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.projection(decoder_output)


if __name__ == "__main__":
    import torch
    model = Transformer(
        source_vocab_size=3000,
        target_vocab_size=3000,
        source_seq_len=155,
        target_seq_len=155,
        d_model=512
    )

    src = torch.randint(0, 3000, (2, 155))
    tgt = torch.randint(0, 3000, (2, 155))
    src_mask = torch.ones(1, 1, 155, 155).int()
    tgt_mask = torch.ones(1, 1, 155, 155).int()

    out = model(src, tgt, src_mask, tgt_mask)
    print("Transformer output shape:", out.shape)  # [2, 155, 3000]
