import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
import os

# Load tokenizer files
tokenizer_en = Tokenizer.from_file("./tokenizer_en/tokenizer_en.json")
tokenizer_ur = Tokenizer.from_file("./tokenizer_ur/tokenizer_ur.json")

# Load max sequence length
with open("./urdugpt/max_seq_len.txt", "r") as f:
    max_seq_len = int(f.read().strip())

# Constants
CLS_ID = tokenizer_ur.token_to_id("[CLS]")
SEP_ID = tokenizer_ur.token_to_id("[SEP]")
PAD_ID = tokenizer_ur.token_to_id("[PAD]")

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def pad_to_len(tensor, length, pad_value):
    if len(tensor) < length:
        return torch.cat([tensor, torch.tensor([pad_value] * (length - len(tensor)))])
    return tensor[:length]

class EncodeDataset(Dataset):
    def __init__(self, raw_dataset, max_seq_len):
        self.raw_dataset = raw_dataset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        data = self.raw_dataset[index]
        source_text = data['translation']['en']
        target_text = data['translation']['ur']

        source_encoded = tokenizer_en.encode(source_text).ids
        target_encoded = tokenizer_ur.encode(target_text).ids

        if not source_encoded or not target_encoded:
            source_encoded = [tokenizer_en.token_to_id("[UNK]")]
            target_encoded = [tokenizer_ur.token_to_id("[UNK]")]

        encoder_input = torch.cat([
            torch.tensor([CLS_ID]),
            torch.tensor(source_encoded),
            torch.tensor([SEP_ID])
        ], dim=0)

        decoder_input = torch.cat([
            torch.tensor([CLS_ID]),
            torch.tensor(target_encoded)
        ], dim=0)

        target_label = torch.cat([
            torch.tensor(target_encoded),
            torch.tensor([SEP_ID])
        ], dim=0)

        encoder_input = pad_to_len(encoder_input, self.max_seq_len, PAD_ID)
        decoder_input = pad_to_len(decoder_input, self.max_seq_len, PAD_ID)
        target_label  = pad_to_len(target_label,  self.max_seq_len, PAD_ID)

        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "target_label": target_label,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "source_text": source_text,
            "target_text": target_text
        }

# Load datasets
print("📦 Loading Urdu datasets (larger sample)...")
train_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ur", split="train[:10000]")
val_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ur", split="validation[:200]")

# Create datasets and dataloaders
train_ds = EncodeDataset(train_dataset, max_seq_len)
val_ds = EncodeDataset(val_dataset, max_seq_len)

train_dataloader = DataLoader(train_ds, batch_size=5, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=1)

print("✅ UrduGPT Dataloaders ready.")
print(f"Train samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
