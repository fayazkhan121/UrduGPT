import torch
from tokenizers import Tokenizer
from khanggpt_step8_transformer import Transformer
from urdugpt_utils import load_config
from urdugpt_step2_dataloader import causal_mask
import os

# Load config and assets
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_len = int(open(config['data']['max_seq_len_path']).read().strip())
tokenizer_en = Tokenizer.from_file(config['data']['tokenizer_en_path'])
tokenizer_ur = Tokenizer.from_file(config['data']['tokenizer_ur_path'])

# Build model
model = Transformer(
    tokenizer_en.get_vocab_size(),
    tokenizer_ur.get_vocab_size(),
    max_seq_len, max_seq_len,
    d_model=config['model']['d_model'],
    num_layers=config['model']['num_layers'],
    num_heads=config['model']['num_heads'],
    d_ff=config['model']['d_ff'],
    dropout_rate=config['model']['dropout']
).to(device)

ckpt_path = os.path.join(config['training']['checkpoint_dir'], config['training']['checkpoint_name'])
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# Input
test_text = "Hello, how are you?"
print("[Input English]", test_text)

enc = tokenizer_en.encode(test_text)
enc_ids = enc.ids
print("[Token IDs]", enc_ids)

if not enc_ids:
    print("⚠️ Tokenizer failed to encode input. Consider retraining tokenizer with more data.")
    exit()

cls = tokenizer_en.token_to_id("[CLS]")
sep = tokenizer_en.token_to_id("[SEP]")
pad = tokenizer_en.token_to_id("[PAD]")

src_tensor = torch.tensor([cls] + enc_ids + [sep] + [pad]*(max_seq_len - len(enc_ids) - 2), dtype=torch.int64).unsqueeze(0).to(device)
src_mask = (src_tensor != pad).unsqueeze(0).unsqueeze(0).int()

with torch.no_grad():
    encoder_output = model.encode(src_tensor, src_mask)
    print("[Encoder output shape]", encoder_output.shape)

    decoder_input = torch.tensor([[tokenizer_ur.token_to_id("[CLS]")]], device=device)
    for step in range(max_seq_len):
        dec_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        decoder_out = model.decode(decoder_input, encoder_output, src_mask, dec_mask)
        logits = model.projection(decoder_out[:, -1])
        print(f"[Step {step}] Logits shape: {logits.shape}")

        _, next_token = torch.max(logits, dim=-1)
        print("→ Predicted token ID:", next_token.item(), "→ Token:", tokenizer_ur.id_to_token(next_token.item()))

        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == tokenizer_ur.token_to_id("[SEP]"):
            break

    output_ids = decoder_input.squeeze(0).tolist()
    print("[Decoded IDs]", output_ids)
    print("[Final Translation]", tokenizer_ur.decode(output_ids))
