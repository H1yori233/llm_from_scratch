import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import multiprocessing as mp

from tokenizers import Tokenizer
from tokenizers.models import BPE


def tokenize_if_needed(text_file, bin_file, tokenizer):
    if os.path.exists(bin_file):
        return
    if not os.path.exists(text_file):
        print(f"Warning: {text_file} not found")
        return

    # Optional: comment out checkpoint feature for speed
    # temp_file = bin_file + ".temp"
    # progress_file = bin_file + ".progress"
    tokens = []
    # start_line = 0

    # # Resume from checkpoint if exists
    # if os.path.exists(temp_file):
    #     print(f"Found temp file, resuming from where we left off...")
    #     tokens = list(np.fromfile(temp_file, dtype=np.uint32))
    #     with open(progress_file, "r") as pf:
    #         start_line = int(pf.read().strip())
    #     print(f"Resuming from line {start_line}, already have {len(tokens)} tokens")

    print(f"Tokenizing {text_file}...")

    # Count total lines for progress
    total_lines = 0
    with open(text_file, "r") as f:
        for _ in f:
            total_lines += 1

    # Larger batch size for better performance
    batch_size = max(10000, mp.cpu_count() * 2000)  # Dynamic batch size
    batch = []

    with open(text_file, "r") as f:
        pbar = tqdm(total=total_lines, desc="Tokenizing")

        for line_num, line in enumerate(f):
            # if line_num < start_line:
            #     continue

            batch.append(line.strip())

            # Process batch when full
            if len(batch) >= batch_size:
                # Use encode_batch for parallel processing
                encoded_batch = tokenizer.encode_batch(batch)
                for encoded in encoded_batch:
                    tokens.extend(encoded.ids)
                batch = []

                # # Save checkpoint less frequently for speed
                # if (line_num + 1) % 50000 == 0:  # Every 50k lines instead of 10k
                #     np.array(tokens, dtype=np.uint32).tofile(temp_file)
                #     with open(progress_file, "w") as pf:
                #         pf.write(str(line_num + 1))

            pbar.update(1)

        # Process remaining batch
        if batch:
            encoded_batch = tokenizer.encode_batch(batch)
            for encoded in encoded_batch:
                tokens.extend(encoded.ids)

        pbar.close()

    # Save final file directly - avoid intermediate saves for speed
    print(f"Saving {len(tokens)} tokens to {bin_file}...")
    np.array(tokens, dtype=np.uint32).tofile(bin_file)

    # # Clean up temp files
    # if os.path.exists(temp_file):
    #     os.remove(temp_file)
    # if os.path.exists(progress_file):
    #     os.remove(progress_file)

    print(f"Tokenization complete: {bin_file}")


def tokenize_if_needed_fast(text_file, bin_file, tokenizer):
    """Fast version: load all lines into memory, no checkpoints"""
    if os.path.exists(bin_file):
        return
    if not os.path.exists(text_file):
        print(f"Warning: {text_file} not found")
        return

    print(f"Fast tokenizing {text_file}...")

    # Read all lines at once - faster for reasonable file sizes
    with open(text_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(lines)} lines...")

    # Large batch processing - use all lines at once for small datasets
    batch_size = min(len(lines), 50000)  # Process in 50k chunks max
    all_tokens = []

    for i in tqdm(range(0, len(lines), batch_size), desc="Processing batches"):
        batch = lines[i : i + batch_size]
        encoded_batch = tokenizer.encode_batch(batch)

        batch_tokens = []
        for encoded in encoded_batch:
            batch_tokens.extend(encoded.ids)
        all_tokens.extend(batch_tokens)

    # Single write operation
    print(f"Saving {len(all_tokens)} tokens...")
    np.array(all_tokens, dtype=np.uint32).tofile(bin_file)
    print(f"Tokenization complete: {bin_file}")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: load all in memory"
    )
    config = json.load(open(parser.parse_args().config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Setup tokenizer
    tokenizer = Tokenizer(
        BPE.from_file(config["data"]["vocab_file"], config["data"]["merges_file"])
    )

    # Choose tokenization method
    tokenize_func = (
        tokenize_if_needed_fast if parser.parse_args().fast else tokenize_if_needed
    )

    # Tokenize datasets
    tokenize_func(
        config["data"].get("raw_train"), config["data"]["train_data"], tokenizer
    )
    tokenize_func(config["data"].get("raw_val"), config["data"]["val_data"], tokenizer)

    # Load tokenized data
    train_data = np.memmap(config["data"]["train_data"], dtype=np.uint32, mode="r")
    val_data = np.memmap(config["data"]["val_data"], dtype=np.uint32, mode="r")


if __name__ == "__main__":
    # Set environment for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train()
