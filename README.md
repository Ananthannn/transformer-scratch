# Transformer from Scratch 🚀

An attempt to re-create the architecture described in the groundbreaking paper **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017) entirely from scratch using PyTorch. 

This project was heavily inspired by Andrej Karpathy's fantastic video tutorial: **["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY)**

The model has been trained on the **Tiny Shakespeare** dataset to generate Shakespeare-like text.

---

## 🧠 Architecture & Codebase Overview

The codebase is organized into several key modules inside the `src/` directory, each handling a specific part of the Transformer pipeline.

### `src/transformer.py`
The core implementation of the Transformer model. It includes all the essential building blocks:
- **`Embedding` & `PossitionalEncoding`**: Converts input tokens into dense vectors and injects sequence position information using sine/cosine functions.
- **`MultiHeadAttention`**: The core mechanism allowing the model to focus on different parts of the input sequence simultaneously.
- **`FeedForwardNetwork`**: A simple two-layer feed-forward network applied to each position separately.
- **`Encoder` & `Decoder`**: The main stacks. The Encoder processes the input, and the Decoder generates the output (incorporating masked self-attention and cross-attention).
- **`Transformer`**: The final wrapper model bringing everything together.

### `src/build_token.py`
Handles the creation of the vocabulary. It uses the HuggingFace `tokenizers` library to train a custom Byte-Pair Encoding (BPE) tokenizer on the raw Tiny Shakespeare text, saving it as `tokenizer.json`.

### `src/dataset.py`
A custom PyTorch `Dataset` (`ShakespeareDataset`) that loads the tokenized text and prepares input-target pairs (sequences of length `block_size` shifted by one token) for next-token prediction training.

### `src/train.py`
The training loop. It initializes the `Transformer`, sets up the `CrossEntropyLoss` and `AdamW` optimizer, and iterates over the dataset. It also dynamically generates causal masks to prevent the decoder from "looking ahead" at future tokens. The final weights are saved as `transformer.pt`.

### `src/predict.py`
The inference script. It loads the trained weights (`transformer.pt`) and provides an interactive loop where you can input a prompt. The model generates continuation text token-by-token using temperature-scaled multinomial sampling.

---

## 🛠️ Setup & Installation

Install the project dependencies using pip:

```bash
python -m pip install -r requirements.txt
```

If you want to keep dependencies isolated, create a virtual environment first:

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
Before training, you need to build the tokenizer from the raw data:
```bash
python src/build_token.py
```

### 2. Train the Model
Train the Transformer on the Tiny Shakespeare dataset (CUDA recommended):
```bash
python src/train.py
```

### 3. Generate Text
Once trained, use the prediction script to interact with the model:
```bash
python src/predict.py
```

*Happy modeling!* 🎭
