import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from urdugpt_utils import load_config

from urdugpt_step2_dataloader import EncodeDataset, causal_mask
from urdugpt_step8_transformer import Transformer
from tokenizers import Tokenizer
from datasets import load_dataset

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_len = int(open(config['data']['max_seq_len_path']).read().strip())
tokenizer_en = Tokenizer.from_file(config['data']['tokenizer_en_path'])
tokenizer_ur = Tokenizer.from_file(config['data']['tokenizer_ur_path'])

source_vocab_size = tokenizer_en.get_vocab_size()
target_vocab_size = tokenizer_ur.get_vocab_size()

train_dataset = load_dataset(
    config['data']['dataset_name'],
    config['data']['dataset_config'],
    split='train'
).select(range(config['data']['train_limit']))

val_dataset = load_dataset(
    config['data']['dataset_name'],
    config['data']['dataset_config'],
    split='validation'
).select(range(config['data']['val_limit']))

train_ds = EncodeDataset(train_dataset, max_seq_len)
val_ds = EncodeDataset(val_dataset, max_seq_len)
train_dataloader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=1)

model = Transformer(
    source_vocab_size, target_vocab_size,
    max_seq_len, max_seq_len,
    d_model=config['model']['d_model'],
    num_layers=config['model']['num_layers'],
    num_heads=config['model']['num_heads'],
    d_ff=config['model']['d_ff'],
    dropout_rate=config['model']['dropout']
).to(device)

loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer_ur.token_to_id("[PAD]"),
    label_smoothing=config['training']['label_smoothing']
)
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

def train_urdugpt():
    for epoch in range(config['training']['epochs']):
        model.train()
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            enc_in = batch["encoder_input"].to(device)
            dec_in = batch["decoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)
            target = batch["target_label"].to(device)

            optimizer.zero_grad()
            out = model(enc_in, dec_in, enc_mask, dec_mask)
            loss = loss_fn(out.view(-1, target_vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        ckpt_path = os.path.join(config['training']['checkpoint_dir'], f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        validate_model(epoch+1)

def validate_model(epoch):
    model.eval()
    print(f"\n🔎 Validating after epoch {epoch}...")
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            if idx >= 2: break
            enc_in = batch["encoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            encoder_output = model.encode(enc_in, enc_mask)
            decoder_input = torch.tensor([[tokenizer_ur.token_to_id("[CLS]")]], device=device)
            for _ in range(max_seq_len):
                dec_mask = causal_mask(decoder_input.size(1)).type_as(enc_mask).to(device)
                out = model.decode(decoder_input, encoder_output, enc_mask, dec_mask)
                logits = model.projection(out[:, -1])
                _, next_token = torch.max(logits, dim=-1)
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == tokenizer_ur.token_to_id("[SEP]"):
                    break
            output_text = tokenizer_ur.decode(decoder_input[0].tolist())
            print("\n🔹 Source:", batch["source_text"][0])
            print("🔸 Target:", batch["target_text"][0])
            print("✅ Predicted:", output_text)

if __name__ == "__main__":
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    train_urdugpt()
