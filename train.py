import torch
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

from core.model import TransformerLM
from core.optimizer import get_optimizer

# from core.bpe_tokenizer import BPETokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from core.utils import (
    lr_cosine_schedule,
    cross_entropy,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


def log_results(config, results):
    """Just dump everything to a markdown table"""
    log_file = config["system"]["log_path"]
    os.makedirs("data", exist_ok=True)

    # flatten everything into one dict
    data = {"time": datetime.now().strftime("%m-%d %H:%M")}
    for k, v in config.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                data[f"{k}_{k2}"] = v2
        else:
            data[k] = v
    data.update(results)

    # load old data if exists
    df = pd.DataFrame([data])
    if os.path.exists(log_file):
        old = (
            pd.read_csv(log_file.replace(".md", ".csv"))
            if os.path.exists(log_file.replace(".md", ".csv"))
            else pd.DataFrame()
        )
        if len(old) > 0:
            df = pd.concat([old, df], ignore_index=True)

    # save both csv and markdown
    csv_file = log_file.replace(".md", ".csv")
    df.to_csv(csv_file, index=False)

    with open(log_file, "w") as f:
        f.write("# Experiments\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write(f"\n\n_Last run: {data['time']}_\n")


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

    # make model with attention choice
    m = config["model"]
    t = config["training"]
    use_flash = t.get("use_flash_attention", True)

    model = TransformerLM(
        vocab_size=m["vocab_size"],
        context_length=m["context_length"],
        d_model=m["d_model"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        d_ff=m["d_ff"],
        rope_theta=m["rope_theta"],
        use_flash_attn=use_flash,
        device=device,
    ).to(device)

    if config["system"].get("use_compile"):
        model = torch.compile(model)

    params = sum(p.numel() for p in model.parameters())
    attn_type = "flash" if use_flash else "standard"
    print(
        f"{config['experiment_name']}: {params/1e6:.1f}M params, {attn_type} attention"
    )

    # choose optimizer
    optimizer_name = t.get("optimizer", "adamw")
    opt = get_optimizer(
        optimizer_name,
        model.parameters(),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )

    print(f"Using {optimizer_name} optimizer")

    # try to resume from checkpoint
    start_step = 0
    if os.path.exists(config["system"]["checkpoint_path"]):
        print(f"Resuming from {config['system']['checkpoint_path']}")
        start_step = load_checkpoint(config["system"]["checkpoint_path"], model, opt)

    best_loss = 999
    best_step = 0
    losses = []

    for step in tqdm(
        range(start_step, t["num_steps"]), desc=config["experiment_name"][:20]
    ):
        # lr schedule
        lr = lr_cosine_schedule(
            step,
            t["learning_rate"],
            t["learning_rate"] * t["min_lr_ratio"],
            t["warmup_steps"],
            t["num_steps"],
        )
        for g in opt.param_groups:
            g["lr"] = lr

        # eval
        if step % t["eval_interval"] == 0:
            model.eval()
            val_loss = 0
            for _ in range(t["eval_batches"]):
                x, y = get_batch(val_data, t["batch_size"], m["context_length"], device)
                with torch.no_grad():
                    val_loss += cross_entropy(
                        model(x).view(-1, m["vocab_size"]), y.view(-1)
                    ).item()
            val_loss /= t["eval_batches"]

            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
                save_checkpoint(model, opt, step, config["system"]["checkpoint_path"])

            model.train()

        # train step
        x, y = get_batch(train_data, t["batch_size"], m["context_length"], device)
        loss = cross_entropy(model(x).view(-1, m["vocab_size"]), y.view(-1))
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), t["grad_clip"])
        opt.step()

    # done
    results = {
        "best_val": best_loss,
        "best_step": best_step,
        "final_train": np.mean(losses[-50:]),
        "params_M": params / 1e6,
        "tokens_M": (t["num_steps"] * t["batch_size"] * m["context_length"]) / 1e6,
        "attention_type": attn_type,
        "optimizer_used": optimizer_name,
    }

    log_results(config, results)
    print(f"Done! Best: {best_loss:.3f} @ {best_step}")


if __name__ == "__main__":
    train()
