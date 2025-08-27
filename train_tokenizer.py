import json
import os
import time
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(input_file, vocab_size=32000, special_tokens=["<|endoftext|>"]):
    """Train BPE tokenizer and save vocab/merges files"""
    print(f"Training tokenizer on {input_file} with vocab_size={vocab_size}")

    # setup
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    initial_tokens = special_tokens + [chr(i) for i in range(256)]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=initial_tokens)

    # train
    start = time.time()
    tokenizer.train(files=[input_file], trainer=trainer)
    hours = (time.time() - start) / 3600

    # output paths
    base = os.path.splitext(os.path.basename(input_file))[0]
    dir = os.path.dirname(input_file)
    vocab_file = os.path.join(dir, f"{base}-vocab_size_{vocab_size}-vocab.json")
    merges_file = os.path.join(dir, f"{base}-vocab_size_{vocab_size}-merges.txt")

    # save temp file and extract data
    temp_file = "temp_tokenizer.json"
    tokenizer.save(temp_file)

    with open(temp_file, "r") as f:
        data = json.load(f)

    # save vocab
    vocab = data["model"]["vocab"]
    with open(vocab_file, "w") as f:
        json.dump(vocab, f, indent=2)

    # save merges
    merges = data["model"]["merges"]
    with open(merges_file, "w") as f:
        for pair in merges:
            f.write(f"{pair[0]} {pair[1]}\n")

    # cleanup
    os.remove(temp_file)

    # stats
    longest = max(vocab.keys(), key=len)
    print(f"Done in {hours:.2f}h")
    print(f"Vocab: {len(vocab)} tokens, longest: '{longest}'")
    print(f"Saved: {vocab_file}")
    print(f"Saved: {merges_file}")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer with HuggingFace")
    parser.add_argument("input_file", help="Text file to train on")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        "--special_tokens",
        nargs="+",
        default=["<|endoftext|>"],
        help="Special tokens to add",
    )
    args = parser.parse_args()

    train_tokenizer(args.input_file, args.vocab_size, args.special_tokens)


if __name__ == "__main__":
    main()
