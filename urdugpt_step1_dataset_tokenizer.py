import os
import torch
from datasets import load_dataset
from torch.utils.data import random_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# UrduGPT paths
os.makedirs("./urdugpt", exist_ok=True)
os.makedirs("./tokenizer_en", exist_ok=True)
os.makedirs("./tokenizer_ur", exist_ok=True)

print("Loading dataset...")
train_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ur", split='train')
validation_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ur", split='validation')

raw_train_dataset, _ = random_split(train_dataset, [1500, len(train_dataset) - 1500])
raw_validation_dataset, _ = random_split(validation_dataset, [50, len(validation_dataset) - 50])

def get_ds_iterator(dataset, lang):
    for data in dataset:
        yield data['translation'][lang]

print("Training English tokenizer...")
tokenizer_en = Tokenizer(BPE(unk_token="[UNK]"))
trainer_en = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], min_frequency=2)
tokenizer_en.pre_tokenizer = Whitespace()
tokenizer_en.train_from_iterator(get_ds_iterator(raw_train_dataset, "en"), trainer=trainer_en)
tokenizer_en.save("./tokenizer_en/tokenizer_en.json")

print("Training Urdu tokenizer...")
tokenizer_ur = Tokenizer(BPE(unk_token="[UNK]"))
trainer_ur = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], min_frequency=2)
tokenizer_ur.pre_tokenizer = Whitespace()
tokenizer_ur.train_from_iterator(get_ds_iterator(raw_train_dataset, "ur"), trainer=trainer_ur)
tokenizer_ur.save("./tokenizer_ur/tokenizer_ur.json")

# Load and calculate max sequence length
tokenizer_en = Tokenizer.from_file("./tokenizer_en/tokenizer_en.json")
tokenizer_ur = Tokenizer.from_file("./tokenizer_ur/tokenizer_ur.json")

max_seq_len_source = 0
max_seq_len_target = 0
for data in raw_train_dataset:
    source_ids = tokenizer_en.encode(data['translation']['en']).ids
    target_ids = tokenizer_ur.encode(data['translation']['ur']).ids
    max_seq_len_source = max(max_seq_len_source, len(source_ids))
    max_seq_len_target = max(max_seq_len_target, len(target_ids))

max_seq_len = max(max_seq_len_source, max_seq_len_target) + 30
with open("./urdugpt/max_seq_len.txt", "w") as f:
    f.write(str(max_seq_len))

print("\n✅ UrduGPT tokenizers ready.")
print(f"English vocab size: {tokenizer_en.get_vocab_size()}")
print(f"Urdu vocab size: {tokenizer_ur.get_vocab_size()}")
print(f"Max sequence length: {max_seq_len}")