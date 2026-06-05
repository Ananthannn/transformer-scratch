from pathlib import Path

import torch
from tokenizers import Tokenizer

from transformer import Transformer

# -----------------------------------
# Device
# -----------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

TOKENIZER_PATH = (
    BASE_DIR.parent
    / "data"
    / "processed"
    / "tokenizer.json"
)

MODEL_PATH = (
    BASE_DIR
    / "transformer.pt"
)

# -----------------------------------
# Tokenizer
# -----------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

VOCAB_SIZE = tokenizer.get_vocab_size()

# -----------------------------------
# Hyperparameters
# MUST MATCH TRAINING
# -----------------------------------

D_MODEL = 128
NUM_HEADS = 4
NUM_NEURONS = 512
NUM_LAYERS = 4

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
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded.")

# -----------------------------------
# Causal Mask
# -----------------------------------

def generate_causal_mask(
    seq_len,
    device
):
    return (
        torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=device
            )
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

# -----------------------------------
# Text Generation
# -----------------------------------

def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8
):

    token_ids = tokenizer.encode(
        prompt
    ).ids

    generated = token_ids.copy()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            x = torch.tensor(
                [generated],
                dtype=torch.long,
                device=device
            )

            T = x.size(1)

            tgt_mask = generate_causal_mask(
                T,
                device
            )

            logits = model(
                x,
                x,
                tgt_mask=tgt_mask
            )

            logits = logits[:, -1, :]

            probs = torch.softmax(
                logits / temperature,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            generated.append(
                next_token.item()
            )

    return tokenizer.decode(
        generated
    )

# -----------------------------------
# Interactive Loop
# -----------------------------------

while True:

    prompt = input(
        "\nPrompt (or 'quit'): "
    )

    if prompt.lower() == "quit":
        break

    output = generate_text(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.8
    )

    print("\nGenerated:\n")
    print(output)