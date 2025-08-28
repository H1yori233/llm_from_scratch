import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse

from tokenizers import Tokenizer
from tokenizers.models import BPE


def tokenize_if_needed(text_file, bin_file, tokenizer):
    if os.path.exists(bin_file):
        return
    if not os.path.exists(text_file):
        print(f"Warning: {text_file} not found")
        return

    print(f"Tokenizing {text_file}...")
    tokens = []
    with open(text_file, "r") as f:
        for line in tqdm(f):
            # tokens.extend(tokenizer.encode(line))
            tokens.extend(tokenizer.encode(line).ids)
    np.array(tokens, dtype=np.uint16).tofile(bin_file)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    config = json.load(open(parser.parse_args().config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # setup tokenizer and data
    # tokenizer = BPETokenizer.from_files(
    #     config["data"]["vocab_file"], config["data"]["merges_file"]
    # )
    tokenizer = Tokenizer(
        BPE.from_file(config["data"]["vocab_file"], config["data"]["merges_file"])
    )
    tokenize_if_needed(
        config["data"].get("raw_train"), config["data"]["train_data"], tokenizer
    )
    tokenize_if_needed(
        config["data"].get("raw_val"), config["data"]["val_data"], tokenizer
    )

    train_data = np.memmap(config["data"]["train_data"], dtype=np.uint16, mode="r")
    val_data = np.memmap(config["data"]["val_data"], dtype=np.uint16, mode="r")


if __name__ == "__main__":
    train()
