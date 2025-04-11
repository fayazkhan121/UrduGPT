import torch
import torch.nn as nn

# 🔁 Feedforward network from Attention paper (2 linear layers with ReLU)
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))

# 📐 Layer Normalization with learnable gamma and beta
class LayerNorm(nn.Module):
    def __init__(self, d_model: int = 512, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# ➕ Add & Norm block: applies skip connection + layer norm
class AddAndNorm(nn.Module):
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm()

    def forward(self, input_tensor, sublayer_output):
        return self.layer_norm(input_tensor + self.dropout(sublayer_output))

if __name__ == "__main__":
    ff = FeedForward(512, 2048)
    ln = LayerNorm()
    addnorm = AddAndNorm()

    dummy_input = torch.rand(2, 155, 512)
    output = addnorm(dummy_input, ff(dummy_input))
    print("Output shape:", output.shape)  # Should be [2, 155, 512]
