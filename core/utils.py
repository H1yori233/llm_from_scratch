import os
from typing import BinaryIO, IO
import numpy.typing as npt
from collections.abc import Iterable
import torch
from torch import Tensor
from jaxtyping import Float, Int
import math
import numpy as np


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Compute the softmax of a tensor.
    """
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    Compute the cross entropy loss.
    """

    max_logits = torch.max(inputs, dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_logits), dim=-1))
    selected = inputs[torch.arange(inputs.shape[0]), targets]
    return torch.mean(log_sum_exp - selected + max_logits)


def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    Return the learning rate according to the scheduler.
    t:          int current iteration
    alpha_max:  float maximum learning rate
    alpha_min:  float minimum learning rate
    T_w:        int warmup period
    T_c:        int cosine decay period
    """
    
    if t < T_w:
        return t / T_w * alpha_max
    elif t < T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (
            alpha_max - alpha_min
        )
    else:
        return alpha_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    """

    grads = [p.grad for p in parameters if p is not None and p.grad is not None]
    if len(grads) == 0:
        return

    l2_norm = torch.norm(torch.cat([g.reshape(-1) for g in grads]))
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + 1e-6)  # Add epsilon for numerical stability
        for g in grads:
            g.mul_(scale)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """

    # random sample the position
    ix = np.random.randint(dataset.size - context_length, size=batch_size)

    # get the input and target
    x = torch.stack(
        [torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in ix]
    )  # (batch_size, context_length)
    y = torch.stack(
        [
            torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64))
            for i in ix
        ]
    )  # (batch_size, context_length)

    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.
    """

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
