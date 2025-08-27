import torch
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Float, Int
from .layers import (
    Linear,
    Embedding,
    RMSNorm,
    SwiGLU,
    RotaryPositionalEmbedding,
    MultiheadSelfAttention,
    FlashMultiheadSelfAttention,
)
from typing import List, Optional
from .utils import softmax


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        rope: RotaryPositionalEmbedding | None = None,
        use_flash_attn: bool = True,
        device=None,
        dtype=None,
    ):
        """
        d_model:        int Dimensionality of the Transformer block inputs.
        num_heads:      int Number of heads to use in multi-head self-attention.
        d_ff:           int Dimensionality of the position-wise feed-forward inner layer.
        rope:           RotaryPositionalEmbedding | None = None RoPE module to use.
        use_flash_attn: bool = True Whether to use Flash Attention or standard attention.
        device:         torch.device | None = None Device to store the parameters on
        dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.use_flash_attn = use_flash_attn
        self.device = device
        self.dtype = dtype

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        # Choose attention type
        if use_flash_attn:
            self.attn = FlashMultiheadSelfAttention(
                d_model, num_heads, rope=rope, device=device, dtype=dtype
            )
        else:
            self.attn = MultiheadSelfAttention(
                d_model, num_heads, rope=rope, device=device, dtype=dtype
            )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """

        if token_positions is None:
            seq_len = x.shape[-2]  # (batch sequence_length d_model)
            token_positions = torch.arange(seq_len, device=x.device)

        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_flash_attn: bool = True,
        device=None,
        dtype=None,
    ):
        """
        vocab_size:     int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        d_model:        int Dimensionality of the Transformer block inputs.
        num_layers:     int The number of Transformer blocks to use.
        num_heads:      int Number of heads to use in multi-head self-attention.
        d_ff:           int Dimensionality of the position-wise feed-forward inner layer.
        rope_theta:     float The RoPE $\\Theta$ parameter.
        use_flash_attn: bool = True Whether to use Flash Attention or standard attention.
        device:         torch.device | None = None Device to store the parameters on
        dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_flash_attn = use_flash_attn
        self.device = device
        self.dtype = dtype

        # create rope
        rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length, device=device
        )
        # rope = None

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope=rope,
                    use_flash_attn=use_flash_attn,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size sequence_length)
        and return a tensor of shape(batch_size sequence_length vocab_size).
        """

        # (batch sequence_length) -> (batch sequence_length d_model)
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_tokens: List[int],
    max_new_tokens: int,
    stop_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    """
    Generate text from a model given a prompt.

    Args:
        model:              TransformerLM The language model.
        prompt_tokens:      List[int] The initial sequence of token IDs (prompt).
        max_new_tokens:     int The maximum number of new tokens to generate.
        stop_token_id:      Optional[int] The token ID that signals the end of generation.
        temperature:        float The temperature for softmax scaling ( > 0). Lower is more deterministic.
        top_p:              float The nucleus sampling threshold (0 < top_p <= 1).

    Returns:
        List[int]: The list of generated token IDs, not including the prompt.
    """

    model.eval()  # switch to eval mode
    device = next(model.parameters()).device
    context_length = model.context_length

    # convert prompt to a tensor with a batch dimension.
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = (
            idx if idx.size(1) <= context_length else idx[:, -context_length:]
        )  # crop the context

        # forward pass
        logits = model(idx_cond)  # (batch_size, context_length, vocab_size)
        logits = logits[:, -1, :]  # (batch_size, 1, vocab_size)

        if temperature > 0:
            logits = logits / temperature

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

            # remove tokens above the threshold.
            sorted_indices_to_remove = cumulative_probs > top_p
            # shift to keep the 1st token.
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[..., indices_to_remove] = float("-inf")  # -inf, prob would be 0

        probs = softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token_id), dim=1)

        if stop_token_id is not None and next_token_id.item() == stop_token_id:
            break

    return idx[0, len(prompt_tokens) :].tolist()  # only new generated tokens
