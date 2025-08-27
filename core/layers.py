import math
import torch
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Float, Int
from .flash_attention import TritonFlashAttentionAutogradFunction


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        in_features:    int final dimension of the input
        out_features:   int final dimension of the output
        device:         torch.device | None = None Device to store the parameters on
        dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        weight = torch.empty(
            self.out_features, self.in_features, device=device, dtype=dtype
        )
        sigma = math.sqrt(2 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(
            weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

        # construct and store parameter as W (not W^T) for memory ordering reasons,
        # putting it in an nn.Parameter
        self.weight = torch.nn.Parameter(weight)
        # we do not include a bias term, following most modern LLMs.
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """

        # x: (..., d_in) | w: (d_out, d_in)
        # ... and d_out keep at right side of ->, so they are public dimension
        # einsum will calculate sum over **d_in** dimension
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim:  int Dimension of the embedding vectors, i.e., d_model
        device:         torch.device | None = None Device to store the parameters on
        dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = torch.empty(
            self.num_embeddings, self.embedding_dim, device=device, dtype=dtype
        )  # (vocab_size, d_model)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

        # initialize embedding matrix as a nn.Parameter
        self.weight = torch.nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """

        # select the embedding vector for each token ID by
        # indexing into an embedding matrix of shape (vocab_size, d_model) using a
        # torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model:    int Hidden dimension of the model
        eps:        float = 1e-5 Epsilon value for numerical stability
        device:     torch.device | None = None Device to store the parameters on
        dtype:      torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        weight = torch.ones(self.d_model, device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (`batch_size`, `sequence_length`, `d_model`)
        and return a tensor of the same shape.
        """

        # upcast input to torch.float32 to prevent overflow when square the input.
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            (1.0 / self.d_model) * torch.sum(x**2, dim=-1, keepdim=True) + self.eps
        )
        result = (x / rms) * self.weight  # g_i

        # Return the result in the original dtype
        return result.to(in_dtype)


class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FFNSiLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)))


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """
        d_model:    int Hidden dimension of the model
        d_ff:       int Dimensionality of the up-project
        device:     torch.device | None = None Device to store the parameters on
        dtype:      torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the SwiGLU output
        """
        w1_x = self.w1(x)
        silu = SiLU()(w1_x)
        w3_x = self.w3(x)

        # Gated Linear Units, using element-wise multiplication
        # reduce the vanishing gradient problem for deep architectures
        # by providing a linear path for the gradients while retaining non-linear capabilities
        glu = silu * w3_x

        return self.w2(glu)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta:          float Θ value for the RoPE
        d_k:            int dimension of query and key vectors
        max_seq_len:    int Maximum sequence length that will be inputted
        device:         torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for Rotary Positional Embedding.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # theta^(2k / d_k) -> 0, 2, ..., d_k-2
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d_k, 2, device=self.device).float() / self.d_k)
        )
        t = torch.arange(self.max_seq_len, device=self.device)
        angles = torch.outer(t, freqs)

        # init with self.register_buffer(persistent=False), instead of a nn.Parameter
        #   (because we do not want to learn these fixed cosine and sine values)
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        """
        Process an input tensor of shape (..., seq_len, d_k)
        and return a tensor of the same shape.
        """

        in_dtype = in_query_or_key.dtype
        sin = self.sin_cached[token_positions].to(
            in_query_or_key.device, dtype=in_dtype
        )
        cos = self.cos_cached[token_positions].to(
            in_query_or_key.device, dtype=in_dtype
        )

        # from (..., seq_len, d_k) to (..., seq_len, d_k // 2, 2)
        x_reshaped = in_query_or_key.reshape(*in_query_or_key.shape[:-1], -1, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]  # get pair

        # apply rotate
        x1_ = cos * x1 - sin * x2  # (..., seq_len, d_k // 2)
        x2_ = sin * x1 + cos * x2  # (..., seq_len, d_k // 2)
        x_reshaped = torch.stack([x1_, x2_], dim=-1)  # (..., seq_len, d_k // 2, 2)
        return x_reshaped.flatten(-2)  # ( ... sequence_length d_k)


class Softmax(torch.nn.Module):
    def __init__(self, dim):
        """
        dim:            int dimension to apply softmax to
        """
        super().__init__()
        self.dim = dim

    def forward(self, in_features) -> torch.Tensor:
        """
        Apply the softmax to the input.
        """

        in_features -= torch.max(in_features, dim=self.dim, keepdim=True)[0]
        exp = torch.exp(in_features)
        sum = torch.sum(exp, dim=self.dim, keepdim=True)
        return exp / sum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.
    """
    d_k = Q.shape[-1]
    scores = einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) / math.sqrt(d_k)

    # adding a −infinity in any entry of the mask matrix that is False
    if mask is not None:
        mask = mask.to(scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))

    softmax = Softmax(dim=-1)
    atten_weight = softmax(scores)
    attention = einsum(
        atten_weight, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )
    return attention


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        """
        d_model:        int Dimensionality of the Transformer block inputs.
        num_heads:      int Number of heads to use in multi-head self-attention.
        rope:           RotaryPositionalEmbedding | None = None RoPE module to use.
        device:         torch.device | None = None Device to store the parameters on
        dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.device = device
        self.dtype = dtype

        # d_k = d_v = d_model/h
        d_k, d_v = d_model // num_heads, d_model // num_heads

        # the learnable parameters
        self.q_proj = Linear(d_model, d_k * num_heads, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_k * num_heads, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_v * num_heads, device=device, dtype=dtype)
        self.output_proj = Linear(d_v * num_heads, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # put last dimension from (h * d_k) to (h, d_k)
        q = rearrange(q, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        k = rearrange(k, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        v = rearrange(v, "... n (h d_v) -> ... h n d_v", h=self.num_heads)

        # RoPE should be applied to the query and key vectors, but not the value vectors.
        # precisely the same RoPE rotation should be applied to the query and key vectors for each head.
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # use torch.triu to construct causal attention mask
        # which allows token i to attend to all positions j ≤ i in the sequence
        # we want to use ~mask to mask out the attention weights and default float type cant be applied
        mask = torch.tril(
            torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool)
        )  # (... queries keys)
        attention = scaled_dot_product_attention(q, k, v, mask)

        # concat multi-head result, put last dimension from (h, d_k) to (h * d_k)
        attention = rearrange(attention, "... h n d_v -> ... n (h d_v)")
        return self.output_proj(attention)


class FlashMultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        """
        Flash Attention implementation of multi-head self-attention.

        d_model:        int Dimensionality of the Transformer block inputs.
        num_heads:      int Number of heads to use in multi-head self-attention.
        rope:           RotaryPositionalEmbedding | None = None RoPE module to use.
        device:         torch.device | None = None Device to store the parameters on
        dtype:          torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.device = device
        self.dtype = dtype

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        # the learnable parameters - project to full d_model for all heads combined
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, sequence_length, d_model)
        token_positions: (sequence_length,) or (batch_size, sequence_length)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)

        # Reshape for multi-head: (batch, seq_len, num_heads, d_k)
        q = rearrange(q, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        k = rearrange(k, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        v = rearrange(v, "... n (h d_v) -> ... h n d_v", h=self.num_heads)

        # Apply RoPE if provided
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Reshape back for flash attention: (batch, seq_len, d_model)
        q = rearrange(q, "... h n d_k -> ... n (h d_k)")
        k = rearrange(k, "... h n d_k -> ... n (h d_k)")
        v = rearrange(v, "... h n d_v -> ... n (h d_v)")

        # Apply flash attention with causal masking
        attention = TritonFlashAttentionAutogradFunction.apply(
            q, k, v, True  # is_causal=True for autoregressive models
        )

        # Apply output projection
        return self.output_proj(attention)
