import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse

from tokenizers import Tokenizer
from tokenizers.models import BPE


def tokenize_if_needed(text_file, bin_file, tokenizer):
    # Skip if already done
    if os.path.exists(bin_file):
        print(f"Binary file exists: {bin_file}")
        return
    if not os.path.exists(text_file):
        print(f"Warning: {text_file} not found")
        return

    print(f"Tokenizing {text_file}...")

    temp_file = bin_file + ".temp"
    checkpoint_file = bin_file + ".checkpoint"

    start_line, total_tokens = 0, 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as ckpt:
            data = json.load(ckpt)
            start_line, total_tokens = data["line"], data["tokens"]
        print(f"Resuming from line {start_line}, {total_tokens} tokens")

    batch_size = 10000  # lines per batch
    write_interval = 1000000  # lines per checkpoint

    with (
        open(text_file, "r") as f,
        open(temp_file, "ab" if start_line else "wb") as out,
    ):
        total_lines = sum(1 for _ in open(text_file, "r"))
        pbar = tqdm(total=total_lines, initial=start_line, desc="Tokenizing")

        batch, line_num = [], 0
        for line_num, line in enumerate(f):
            if line_num < start_line:
                continue

            batch.append(line.strip())
            if len(batch) >= batch_size:
                encoded = tokenizer.encode_batch(batch)
                flat = np.array([id for e in encoded for id in e.ids], dtype=np.uint32)
                flat.tofile(out)
                total_tokens += len(flat)
                batch.clear()
                pbar.update(batch_size)

            if (line_num + 1) % write_interval == 0:
                out.flush()
                with open(checkpoint_file, "w") as ckpt:
                    json.dump({"line": line_num + 1, "tokens": total_tokens}, ckpt)
                tqdm.write(f"Checkpoint at line {line_num+1}: {total_tokens} tokens")

        if batch:
            encoded = tokenizer.encode_batch(batch)
            flat = np.array([id for e in encoded for id in e.ids], dtype=np.uint32)
            flat.tofile(out)
            total_tokens += len(flat)
            pbar.update(len(batch))

        pbar.close()

    os.rename(temp_file, bin_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print(f"Done: {bin_file} ({total_tokens} tokens)")


def train():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Config file path")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup tokenizer
    tokenizer = Tokenizer(
        BPE.from_file(config["data"]["vocab_file"], config["data"]["merges_file"])
    )

    # Tokenize datasets
    tokenize_if_needed(
        config["data"].get("raw_train"), config["data"]["train_data"], tokenizer
    )

    tokenize_if_needed(
        config["data"].get("raw_val"), config["data"]["val_data"], tokenizer
    )


if __name__ == "__main__":
    # Enable parallel tokenization
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train()
