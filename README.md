# UrduGPT Web App (Streamlit)

This is the official UI for **UrduGPT** — a custom-built English → Urdu translator powered by a Transformer-based LLM trained from scratch using PyTorch.
![Screenshot 2025-04-11 101730](https://github.com/user-attachments/assets/79b236a2-d9c8-424c-84dc-98c800adda86)

---

## 🧠 What is UrduGPT?
UrduGPT is a research and production-friendly language model built step-by-step using:
- Raw dataset from Hugging Face (English–Urdu parallel corpus)
- Byte-Pair Encoding (BPE) tokenizers trained from scratch
- Transformer architecture inspired by "Attention Is All You Need"
- PyTorch for model building & training
- Streamlit for live translation web app

---

## ✅ Features
- Sentence translation with beam or greedy decoding
- Token-by-token display with confidence scores
- Translation history session & export (CSV, PDF, Word)
- Local branding (logo, favicon)
- Deployable on Streamlit Cloud or Hugging Face Spaces

---

## 🛠️ Step-by-Step Model Building

### 🔹 Step 1: Load Dataset
```bash
python urdugpt_step1_dataset.py
```
Loads and trims parallel corpus (English–Urdu) from Hugging Face.

### 🔹 Step 2: Train Tokenizer
```bash
python urdugpt_step2_tokenizer.py
```
Trains BPE tokenizers for both source (English) and target (Urdu) languages.

### 🔹 Step 3: Prepare Dataloaders
```bash
python urdugpt_step2_dataloader.py
```
Creates PyTorch-compatible dataset & dataloaders.

### 🔹 Step 4–8: Transformer Model
Model code is inside `urdugpt_step8_transformer.py`, built from scratch:
- Embedding + Positional Encoding
- Multi-head Attention
- FeedForward + AddNorm
- Encoder and Decoder stacks
- Final Projection layer

### 🔹 Step 9: Train Model
```bash
python urdugpt_step9_train.py
```
- Uses cross-entropy loss
- Trains for N epochs and saves checkpoints (./urdugpt/model_{epoch}.pt)

### 🔹 Step 10: Translate (CLI)
```bash
python urdugpt_translate.py
```
Interactive terminal-based translation using latest model checkpoint.

### 🔹 Step 11: Streamlit UI
```bash
streamlit run urdugpt_web_app.py
```
Clean web-based frontend with history, export, and visual confidence scores.

---

## 📦 Installation
```bash
pip install -r requirements.txt
```

---

## 🚀 Run Locally
Make sure the model is trained and tokenizer files exist.
Then run:
```bash
streamlit run urdugpt_web_app.py
```

---

## 🌐 Deploy to Streamlit Cloud
1. Clone this to a public GitHub repository
2. Go to https://streamlit.io/cloud
3. Click **New App** → select your repo → `urdugpt_web_app.py`
4. Set Python version and add `requirements.txt`
5. Hit **Deploy** 🎉

---

## 🤝 Open Source Plans
This project will be used:
- To demonstrate building LLMs from scratch
- As a template for multilingual translation apps
- To support fine-tuning for Urdu/Indic NLP research

We’ll invite contributors to:
- Extend to other language pairs (e.g., English → Pashto, Hindi, Bangali, Panjabi)
- Improve UI/UX (add voice input, transliteration)
- Add dataset upload & training interface

---

## 💡 Benefits
- Learn Transformer internals end-to-end
- Translate with your own trained model (no API needed)
- Run entirely offline or host on open platforms
- Extendable to many other NLP tasks

---

## 📁 Files
```
urdugpt_web_app.py       # Streamlit UI
urdugpt_utils.py          # Config loader (YAML)
urdugpt_step1_dataset.py
urdugpt_step2_tokenizer.py
urdugpt_step2_dataloader.py
urdugpt_step8_transformer.py
urdugpt_step9_train.py
urdugpt_translate.py
config.yaml               # All hyperparameters & paths
favicon.ico               # UI icon
urdugpt.png               # Logo for UI
```

---

## 🙌 Author

This project is proudly created and maintained by **Fayaz Khan**.

- 🔗 [GitHub](https://github.com/fayazkhan121)
- 💼 [LinkedIn](https://www.linkedin.com/in/contact-fayaz-khan/)

## 🧾 License
MIT License. Contributions welcome.

Made with ❤️ for Urdu speakers & NLP builders.
