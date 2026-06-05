# Transformer from Scratch 🚀
A full re-creation of the architecture described in the groundbreaking paper **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017), built entirely from scratch using PyTorch — trained on the **Tiny Shakespeare** dataset to generate Shakespeare-like text.

Heavily inspired by Andrej Karpathy's fantastic tutorial: **["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY)**

---

## 🧠 Features

- **Custom Tokenizer:** Trains a Byte-Pair Encoding (BPE) tokenizer from scratch on raw text using HuggingFace `tokenizers`.
- **Full Transformer Architecture:** Encoder + Decoder with multi-head attention, positional encoding, and feed-forward layers — all implemented from scratch.
- **Shakespeare Text Generation:** Trained on Tiny Shakespeare with temperature-scaled multinomial sampling for creative text generation.
- **Causal Masking:** Dynamically generated masks prevent the decoder from attending to future tokens during training.

---

## 🖥️ Tech Stack

- **Framework:** Python, PyTorch
- **Tokenizer:** HuggingFace `tokenizers` (BPE)
- **Dataset:** Tiny Shakespeare

---

## 📦 Project Structure

```
transformer-from-scratch/
├── src/
│   ├── transformer.py      # Full Transformer architecture
│   ├── build_token.py      # BPE tokenizer training
│   ├── dataset.py          # PyTorch Dataset for Shakespeare
│   ├── train.py            # Training loop
│   ├── predict.py          # Inference & text generation
│   └── transformer.pt      # Saved model weights
├── data/
│   ├── raw/
│   │   └── raw_data.txt    # Raw Tiny Shakespeare text
│   └── processed/
│       └── tokenizer.json  # Trained BPE tokenizer
└── requirements.txt
```

---

## 🔍 Architecture Overview

### `src/transformer.py`
The core implementation containing all essential building blocks:
- **`Embedding` & `PositionalEncoding`** — Converts tokens into dense vectors and injects positional info via sine/cosine functions.
- **`MultiHeadAttention`** — Allows the model to attend to different parts of the sequence simultaneously.
- **`FeedForwardNetwork`** — A two-layer feed-forward network applied position-wise.
- **`Encoder` & `Decoder`** — The main stacks; the Encoder processes input, the Decoder generates output with masked self-attention and cross-attention.
- **`Transformer`** — The final wrapper bringing all components together.

### `src/build_token.py`
Trains a custom BPE tokenizer on raw Tiny Shakespeare text and saves it as `tokenizer.json`.

### `src/dataset.py`
A custom PyTorch `Dataset` (`ShakespeareDataset`) that loads tokenized text and prepares input-target pairs of length `block_size` (shifted by one token) for next-token prediction.

### `src/train.py`
The training loop — initializes the model, sets up `CrossEntropyLoss` and `AdamW`, iterates over the dataset with dynamic causal masks, and saves final weights to `transformer.pt`.

### `src/predict.py`
Inference script — loads `transformer.pt` and runs an interactive loop where you provide a prompt and the model generates continuation text token-by-token.

---

## 🛠️ Setup & Installation

```bash
python -m pip install -r requirements.txt
```

To keep dependencies isolated, use a virtual environment:

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate

python -m pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Build the Tokenizer
```bash
python src/build_token.py
```

### 2. Train the Model
*(CUDA recommended)*
```bash
python src/train.py
```

### 3. Generate Text
```bash
python src/predict.py
```

---

## 🌐 Connect with Me

[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=flat&logo=instagram&logoColor=white)](https://www.instagram.com/v_ananthann_?igsh=MWFlcHo5a2pvNm5yaA==)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/v-anantha-krishnan-739b942a5/)
[![Email](https://img.shields.io/badge/Email-%23D14836.svg?style=flat&logo=gmail&logoColor=white)](mailto:vananthakrs@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-%2312100E.svg?style=flat&logo=github&logoColor=white)](https://github.com/Ananthannn)

---

## 📄 License

MIT License

---

> Made with ❤️ by V Anantha Krishnan
