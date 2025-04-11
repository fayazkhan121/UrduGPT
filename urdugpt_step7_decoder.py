import torch.nn as nn
from urdugpt_step4_multihead_attention import MultiHeadAttention
from urdugpt_step5_ffn_norm import FeedForward, AddAndNorm, LayerNorm

# 🧱 One decoder block: masked self-attn + cross-attn + feedforward
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.addnorm1 = AddAndNorm(dropout_rate)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.addnorm2 = AddAndNorm(dropout_rate)

        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.addnorm3 = AddAndNorm(dropout_rate)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.addnorm1(x, self.masked_attention(x, x, x, decoder_mask))
        x = self.addnorm2(x, self.cross_attention(x, encoder_output, encoder_output, encoder_mask))
        x = self.addnorm3(x, self.feed_forward(x))
        return x

# 🏗️ Full decoder with stacked decoder blocks
class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.final_norm(x)


if __name__ == "__main__":
    import torch
    decoder = Decoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048)

    dummy_decoder_input = torch.rand(2, 155, 512)
    dummy_encoder_output = torch.rand(2, 155, 512)
    dummy_encoder_mask = torch.ones(1, 1, 155, 155).int()
    dummy_decoder_mask = torch.ones(1, 1, 155, 155).int()

    out = decoder(dummy_decoder_input, dummy_encoder_output, dummy_encoder_mask, dummy_decoder_mask)
    print("Decoder output shape:", out.shape)  # [2, 155, 512]
