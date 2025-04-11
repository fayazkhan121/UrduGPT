import torch.nn as nn
from urdugpt_step4_multihead_attention import MultiHeadAttention
from urdugpt_step5_ffn_norm import FeedForward, AddAndNorm, LayerNorm

# 🧱 One full encoder block (attention + feedforward + normalization)
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.addnorm1 = AddAndNorm(dropout_rate)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.addnorm2 = AddAndNorm(dropout_rate)

    def forward(self, x, mask):
        x = self.addnorm1(x, self.attention(x, x, x, mask))
        x = self.addnorm2(x, self.ffn(x))
        return x

# 🏗️ Full encoder with a stack of 6 encoder blocks
class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)  # normalize at the end too


if __name__ == "__main__":
    import torch
    encoder = Encoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048)

    dummy_input = torch.rand(2, 155, 512)
    dummy_mask = torch.ones(1, 1, 155, 155).int()
    out = encoder(dummy_input, dummy_mask)
    print("Encoder output shape:", out.shape)  # [2, 155, 512]
