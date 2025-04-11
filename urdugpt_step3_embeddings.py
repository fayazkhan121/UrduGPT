import torch
import torch.nn as nn
import math

# 🔤 Embedding layer to map token IDs to dense vectors
class EmbeddingLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids):
        # Multiply by sqrt(d_model) to scale as per "Attention is All You Need"
        return self.embedding(input_ids) * math.sqrt(self.d_model)

# 📐 Positional Encoding using sinusoidal functions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # sin on even indices, cos on odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
    
if __name__ == "__main__":
    emb = EmbeddingLayer(d_model=512, vocab_size=1000)
    pos = PositionalEncoding(d_model=512, max_seq_len=155)

    dummy_input = torch.randint(0, 1000, (2, 155))  # batch_size=2, seq_len=155
    embedded = emb(dummy_input)
    output = pos(embedded)
    print("Output shape:", output.shape)  # Should be [2, 155, 512]
