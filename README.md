# LLM From Scratch

![cover](./data/llm-from-scratch.png)

This project implements a full transformer language model without relying on high-level frameworks like HuggingFace or PyTorch's built-in attention. Every component is built from scratch.

This repository simply combines my solutions for [assignment 1](https://github.com/stanford-cs336/assignment1-basics) and [assignment 2](https://github.com/stanford-cs336/assignment2-systems) of [Stanford CS336](https://stanford-cs336.github.io/spring2025/) course. I’ve merged them into a single repository for convenience, without adding any content beyond the original assignments.




## Features

  * **Hand-Coded Transformer Architecture**: Features RoPE positional encodings, RMSNorm, and SwiGLU FFNs.
  * **Flash Attention with Triton**: A custom implementation with hand-written Triton kernels to optimize GPU memory usage from O(N²) down to O(N).
  * **Multiple Optimizers**: Includes AdamW and SGD, with an extensible design to easily add more.
  * **BPE Tokenizer from Scratch**: With proper handling for special tokens.
  * **A Complete Training System**: Manages experiments with JSON configs and automatically logs results to a Markdown file.



## Quick Start

### 1. Download Data

This project uses data from TinyStories and a subsample of OpenWebText.

```sh
mkdir -p data
cd data

# Download the datasets
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Start Training

Several example configurations are provided to get you started.

```bash
# Default training (Flash Attention + AdamW)
python train.py --config config.json

# Try other configurations
python train.py --config config_large.json
python train.py --config config_std_sgd.json

# View experiment results
cat data/output.md
```



## Experiment Tracking

All experiments are automatically logged with comprehensive metrics:

| timestamp | experiment_name | optimizer | attention_type | best_val_loss | params_M | tokens_M |
|-----------|----------------|-----------|----------------|---------------|----------|----------|
| 12-15 10:30 | baseline_4L_8H | adamw | flash | 2.123 | 25.6 | 163.8 |
| 12-15 14:20 | large_6L_16H | adamw | flash | 1.987 | 67.2 | 327.7 |
| 12-15 16:45 | std_attention | sgd | standard | 2.345 | 25.6 | 163.8 |



## Extending the System

Adding new optimizers is straightforward:
```python
class NewOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4):
        # Implementation here
        
    def step(self, closure=None):
        # Update logic here

# Register in factory
optimizers = {
    "adamw": AdamW,
    "sgd": SGD, 
    "new_optimizer": NewOptimizer,  # Add here
}
```

---

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/H1yori233/llm-from-scratch)

