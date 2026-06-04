from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer


class ShakespeareDataset(Dataset):

    def __init__(
        self,
        text_path,
        tokenizer_path,
        block_size
    ):

        self.block_size = block_size

        # ---------------------
        # Load tokenizer
        # ---------------------

        self.tokenizer = (
            Tokenizer.from_file(
                str(tokenizer_path)
            )
        )

        # ---------------------
        # Read text
        # ---------------------

        with open(
            text_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        # ---------------------
        # Encode entire corpus
        # ---------------------

        encoded = self.tokenizer.encode(
            text
        )

        self.token_ids = encoded.ids

    def __len__(self):

        return (
            len(self.token_ids)
            -
            self.block_size
        )

    def __getitem__(
        self,
        idx
    ):

        x = self.token_ids[
            idx :
            idx + self.block_size
        ]

        y = self.token_ids[
            idx + 1 :
            idx + self.block_size + 1
        ]

        return (
            torch.tensor(
                x,
                dtype=torch.long
            ),
            torch.tensor(
                y,
                dtype=torch.long
            )
        )

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    text_path = (
        project_root /
        "data" /
        "raw" /
        "raw_data.txt"
    )

    tokenizer_path = (
        project_root /
        "data" /
        "processed" /
        "tokenizer.json"
    )

    dataset = ShakespeareDataset(
        text_path=text_path,
        tokenizer_path=tokenizer_path,
        block_size=16
    )

    print(
        "Dataset Size:",
        len(dataset)
    )

    x, y = dataset[0]

    print("Input:")
    print(x)

    print()

    print("Target:")
    print(y)