from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from transformer import Transformer
from training.data.dataset import ShakespeareDataset

# -----------------------------------
# Device
# -----------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

TEXT_PATH = (
    BASE_DIR
    / "training"
    / "data"
    / "raw-data"
    / "raw_data.txt"
)

TOKENIZER_PATH = (
    BASE_DIR
    / "training"
    / "data"
    / "cleaned-data"
    / "tokenizer.json"
)

# -----------------------------------
# Tokenizer
# -----------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

VOCAB_SIZE = tokenizer.get_vocab_size()

print(
    f"Vocabulary Size: {VOCAB_SIZE}"
)

# -----------------------------------
# Hyperparameters
# -----------------------------------

BATCH_SIZE = 16

BLOCK_SIZE = 64

D_MODEL = 128

NUM_HEADS = 4

NUM_NEURONS = 512

NUM_LAYERS = 2

LEARNING_RATE = 3e-4

EPOCHS = 10

# -----------------------------------
# Dataset
# -----------------------------------

dataset = ShakespeareDataset(
    text_path=TEXT_PATH,
    tokenizer_path=TOKENIZER_PATH,
    block_size=BLOCK_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# -----------------------------------
# Model
# -----------------------------------

model = Transformer(
    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_neurons=NUM_NEURONS,
    num_layers=NUM_LAYERS
).to(device)

# -----------------------------------
# Loss
# -----------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------------
# Training Loop
# -----------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for src, target in loader:

        src = src.to(device)

        target = target.to(device)

        # -----------------------------------
        # Encoder Input
        # -----------------------------------

        encoder_input = src

        # -----------------------------------
        # Decoder Input
        # -----------------------------------

        decoder_input = src

        # -----------------------------------
        # Forward
        # -----------------------------------

        logits = model(
            encoder_input,
            decoder_input
        )

        # logits
        # (B,T,V)

        B, T, V = logits.shape

        loss = criterion(
            logits.view(
                B * T,
                V
            ),
            target.view(
                B * T
            )
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss
        / len(loader)
    )

    print(
        f"Epoch {epoch+1} "
        f"Loss: {avg_loss:.4f}"
    )

# -----------------------------------
# Save
# -----------------------------------

torch.save(
    model.state_dict(),
    "transformer.pt"
)

print("Model saved.")