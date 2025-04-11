import torch
from tokenizers import Tokenizer
from urdugpt_step8_transformer import Transformer
from urdugpt_step2_dataloader import causal_mask
from urdugpt_utils import load_config

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_len = int(open(config['data']['max_seq_len_path']).read().strip())
tokenizer_en = Tokenizer.from_file(config['data']['tokenizer_en_path'])
tokenizer_ur = Tokenizer.from_file(config['data']['tokenizer_ur_path'])

def load_model():
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
    return model

def beam_search_decode(model, src_tensor, src_mask, beam_width):
    cls_id = tokenizer_ur.token_to_id("[CLS]")
    sep_id = tokenizer_ur.token_to_id("[SEP]")
    encoder_output = model.encode(src_tensor, src_mask)
    sequences = [[torch.tensor([cls_id], device=device), 0.0]]

    for _ in range(max_seq_len):
        candidates = []
        for seq, score in sequences:
            if seq[-1].item() == sep_id:
                candidates.append((seq, score))
                continue
            tgt_mask = causal_mask(seq.size(0)).type_as(src_mask).to(device)
            out = model.decode(seq.unsqueeze(0), encoder_output, src_mask, tgt_mask)
            logits = model.projection(out[:, -1])
            probs = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(probs, beam_width)
            for i in range(beam_width):
                token = topk.indices[0][i].item()
                new_score = topk.values[0][i].item()
                new_seq = torch.cat([seq, torch.tensor([token], device=device)])
                candidates.append((new_seq, score + new_score))
        sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1].item() == sep_id for seq, _ in sequences):
            break
    return tokenizer_ur.decode(sequences[0][0].tolist())

def translate_text(text, model):
    with torch.no_grad():
        tokens = tokenizer_en.encode(text).ids
        cls = tokenizer_en.token_to_id("[CLS]")
        sep = tokenizer_en.token_to_id("[SEP]")
        pad = tokenizer_en.token_to_id("[PAD]")
        src = torch.tensor([cls] + tokens + [sep] + [pad]*(max_seq_len - len(tokens) - 2), dtype=torch.int64).unsqueeze(0).to(device)
        src_mask = (src != pad).unsqueeze(0).unsqueeze(0).int()
        return beam_search_decode(model, src, src_mask, config['inference']['beam_width'])

if __name__ == "__main__":
    import os
    model = load_model()
    while True:
        text = input("\n📝 Enter English text (or 'exit'): ").strip()
        if text.lower() == "exit":
            break
        print("🇵🇰 UrduGPT says:", translate_text(text, model))
