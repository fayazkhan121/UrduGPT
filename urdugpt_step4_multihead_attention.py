import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout_rate)

        # Learnable weights for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Final projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # Step 1: Linear projection
        batch_size = q.size(0)

        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Step 2: Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled Dot-Product Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # shape: (B, H, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attention_output = attn_weights @ V  # shape: (B, H, T_q, d_k)

        # Step 4: Concatenate heads
        concat = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Step 5: Final linear layer
        output = self.W_o(concat)

        return output
if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    dummy_input = torch.rand(2, 155, 512)  # (batch_size, seq_len, d_model)
    dummy_mask = torch.ones(1, 1, 155, 155).int()  # optional

    out = mha(dummy_input, dummy_input, dummy_input, dummy_mask)
    print("Output shape:", out.shape)  # Should be [2, 155, 512]
