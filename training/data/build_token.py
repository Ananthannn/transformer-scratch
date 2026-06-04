from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

base_dir = Path(__file__).resolve().parent

raw_data_path = (
    base_dir /
    "raw-data" /
    "raw_data.txt"
)

output_dir = (
    base_dir /
    "cleaned-data"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_dir /
    "tokenizer.json"
)

tokenizer = Tokenizer(
    BPE(
        unk_token="[UNK]"
    )
)

tokenizer.pre_tokenizer = (
    Whitespace()
)

trainer = BpeTrainer(
    vocab_size=3000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

if __name__ == "__main__":

    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_data_path}"
        )

    tokenizer.train(
        [str(raw_data_path)],
        trainer
    )

    tokenizer.save(
        str(output_path)
    )

    print(
        f"Tokenizer saved to {output_path}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size()}"
    )

    print(
        "PAD:",
        tokenizer.token_to_id("[PAD]")
    )

    print(
        "BOS:",
        tokenizer.token_to_id("[BOS]")
    )

    print(
        "EOS:",
        tokenizer.token_to_id("[EOS]")
    )

    sample = (
        "First Citizen: "
        "Let us kill him"
    )

    encoded = tokenizer.encode(sample)

    print(encoded.tokens)
    print(encoded.ids)