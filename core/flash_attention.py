import torch
import math
import triton
import triton.language as tl


# Configuration options for Triton autotuning
FLASH_ATTENTION_CONFIGS = [
    triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=3),
    triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_warps=4, num_stages=3),
    triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=8, num_stages=4),
    triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_warps=8, num_stages=4),
    triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 128}, num_warps=8, num_stages=4),
    triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=16, num_stages=4),
]


@triton.autotune(
    configs=FLASH_ATTENTION_CONFIGS,
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,  # (b, (Tr Br), d)
    stride_kb, stride_kk, stride_kd,  # (b, (Tc Bc), d)
    stride_vb, stride_vk, stride_vd,  # (b, (Tc Bc), d)
    stride_ob, stride_oq, stride_od,  # (b, (Tr Br), d)
    stride_lb, stride_lq,  # (b, (Tr Br),)
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # D dimension is major
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    input_dtype = Q_ptr.type.element_ty
    acc_dtype = tl.float32

    # load Q_i to on-chip SRAM
    # since Q_TILE_SIZE might not divide N_QUERIES, and D is fixed, check only the first dim
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r, d)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=acc_dtype)
    L_i = tl.zeros((Q_TILE_SIZE,), dtype=acc_dtype)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=acc_dtype)

    T_c = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_start = query_tile_index * Q_TILE_SIZE
    # skipping all tiles that are always all zero
    max_j = tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE) if is_causal else T_c
    T_c_eff = min(T_c, max_j)
    
    for j in range(T_c_eff):
        k_start = j * K_TILE_SIZE
        
        # load K_j, V_j to on-chip SRAM
        # since K_TILE_SIZE might not divide N_KEYS, and D is fixed, check only the first dim
        K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero") # (B_c, d)
        V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero") # (B_c, d)
        m_prev, l_prev, O_prev = m_i, L_i, O_i

        # keep Q_i and K_j in their original low precision for tl.dot
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (Br, Bc)
        if is_causal and q_start < (k_start + K_TILE_SIZE - 1):  # diagonal case
            q_pos = q_start + tl.arange(0, Q_TILE_SIZE)  # (B_r,)
            k_pos = k_start + tl.arange(0, K_TILE_SIZE)  # (B_c,)
            mask = (
                q_pos[:, None] >= k_pos[None, :]
            )  # (B_r, 1) >= (1, B_c) -> (B_r, B_c)
            S_ij = tl.where(mask, S_ij, -float("inf"))

        m_ij = tl.max(S_ij, axis=1)  # (B_r,)
        m_i = tl.maximum(m_prev, m_ij)  # (B_r,), element-wise maximum

        P_ij = tl.exp(S_ij - m_i[:, None])  # (B_r, B_c)
        exp_m_diff = tl.exp(m_prev - m_i)  # (B_r,)
        L_ij = tl.sum(P_ij, axis=1)  # (B_r,)
        L_i = exp_m_diff * l_prev + L_ij  # (B_r,)
        O_i = tl.dot(
            P_ij.to(V_j.dtype), V_j, acc=exp_m_diff[:, None] * O_prev
        )  # (B_r, d)

        # advance block pointers at the end of the loop.
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = (1.0 / L_i[:, None]) * O_i
    L_i = m_i + tl.log(L_i)

    # Write O_i, L_i to HBM
    #  Since Q_TILE_SIZE might not divide N_QUERIES, and D is fixed, check only the first dim
    tl.store(O_block_ptr, O_i.to(input_dtype), boundary_check=(0,))
    tl.store(L_block_ptr, L_i.to(input_dtype), boundary_check=(0,))



@triton.autotune(
    configs=FLASH_ATTENTION_CONFIGS,
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_backward_dkv_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dK_ptr, dV_ptr, dO_ptr,
    stride_qb, stride_qq, stride_qd,  # (b, (Tr Br), d)
    stride_kb, stride_kk, stride_kd,  # (b, (Tc Bc), d)
    stride_vb, stride_vk, stride_vd,  # (b, (Tc Bc), d)
    stride_ob, stride_oq, stride_od,  # (b, (Tr Br), d)
    stride_lb, stride_lq,  # (b, (Tr Br),)
    stride_db, stride_dd,  # (b, (Tr Br),)
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # D dimension is major
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),  # shared in all tiles
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dd,),
        offsets=(0,),  # shared in all tiles
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    input_dtype = K_ptr.type.element_ty
    acc_dtype = tl.float32

    # load K_j, V_j to on-chip SRAM
    # since K_TILE_SIZE might not divide N_KEYS, and D is fixed, check only the first dim
    K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_c, d)
    V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_c, d)
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=acc_dtype)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=acc_dtype)

    T_r = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    k_start = key_tile_index * K_TILE_SIZE
    min_i = k_start // Q_TILE_SIZE if is_causal else 0
    # advance pointers to the correct starting query tile for causal masking
    if is_causal:
        start_offset_q = min_i * Q_TILE_SIZE
        Q_block_ptr = tl.advance(Q_block_ptr, (start_offset_q, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (start_offset_q, 0))
        L_block_ptr = tl.advance(L_block_ptr, (start_offset_q,))
        D_block_ptr = tl.advance(D_block_ptr, (start_offset_q,))
    
    for i in range(min_i, T_r):
        q_start = i * Q_TILE_SIZE
        
        # load Q_i, dO_i, L_i, D_i to on-chip SRAM
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r, d)
        dO_i = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r, d)
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r,)
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r,)
        
        # keep Q_i and K_j in their original low precision for tl.dot
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (B_r, B_c)
        if is_causal and q_start < (k_start + K_TILE_SIZE - 1):  # diagonal case
            q_pos = q_start + tl.arange(0, Q_TILE_SIZE)  # (B_r,)
            k_pos = k_start + tl.arange(0, K_TILE_SIZE)  # (B_c,)
            mask = q_pos[:, None] >= k_pos[None, :]  # (B_r, B_c)
            S_ij = tl.where(mask, S_ij, -float("inf"))
        
        P_ij = tl.exp(S_ij - L_i[:, None])  # (B_r, B_c)
        dV_j = dV_j + tl.dot(tl.trans(P_ij).to(input_dtype), dO_i)  # (B_c, d)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))  # (B_r, B_c)
        dS_ij = P_ij * (dP_ij - D_i[:, None])  # (B_r, B_c)
        dK_j = dK_j + tl.dot(tl.trans(dS_ij).to(input_dtype), Q_i) * scale  # (B_c, d)

        # advance block pointers at the end of the loop.
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

    # Write dK_j, dV_j to HBM
    tl.store(dK_block_ptr, dK_j.to(input_dtype), boundary_check=(0,))
    tl.store(dV_block_ptr, dV_j.to(input_dtype), boundary_check=(0,))



@triton.autotune(
    configs=FLASH_ATTENTION_CONFIGS,
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_backward_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dQ_ptr, dO_ptr,
    stride_qb, stride_qq, stride_qd,  # (b, (Tr Br), d)
    stride_kb, stride_kk, stride_kd,  # (b, (Tc Bc), d)
    stride_vb, stride_vk, stride_vd,  # (b, (Tc Bc), d)
    stride_ob, stride_oq, stride_od,  # (b, (Tr Br), d)
    stride_lb, stride_lq,  # (b, (Tr Br),)
    stride_db, stride_dd,  # (b, (Tr Br),)
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # D dimension is major
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dd,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    input_dtype = Q_ptr.type.element_ty
    acc_dtype = tl.float32

    Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r, d)
    dO_i = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r, d)
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r,)
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_r,)
    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=acc_dtype)

    T_c = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_start = query_tile_index * Q_TILE_SIZE
    # skipping all tiles that are always all zero
    max_j = tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE) if is_causal else T_c
    T_c_eff = min(T_c, max_j)
    
    for j in range(T_c_eff):
        k_start = j * K_TILE_SIZE
        
        # load K_j, V_j to on-chip SRAM
        K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_c, d)
        V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")  # (B_c, d)
        
        # keep Q_i and K_j in their original low precision for tl.dot
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (B_r, B_c)
        if is_causal and q_start < (k_start + K_TILE_SIZE - 1):  # diagonal case
            q_pos = q_start + tl.arange(0, Q_TILE_SIZE)  # (B_r,)
            k_pos = k_start + tl.arange(0, K_TILE_SIZE)  # (B_c,)
            mask = q_pos[:, None] >= k_pos[None, :]  # (B_r, B_c)
            S_ij = tl.where(mask, S_ij, -float("inf"))

        P_ij = tl.exp(S_ij - L_i[:, None])  # (B_r, B_c)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))  # (B_r, B_c)
        dS_ij = P_ij * (dP_ij - D_i[:, None])  # (B_r, B_c)

        dQ_i = dQ_i + tl.dot(dS_ij.to(input_dtype), K_j) * scale  # (B_r, d)

        # advance block pointers at the end of the loop.
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # atomic write dQ_i, (b, (Tr Br), d)
    tl.store(dQ_block_ptr, dQ_i, boundary_check=(0,))


class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    """
    PyTorch autograd function for Triton-based Flash Attention.
    
    Provides efficient attention computation with O(N) memory complexity
    instead of the standard O(N²) complexity of naive attention.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Compute flash attention forward pass.
        
        Q:          torch.Tensor Query tensor of shape (batch, seq_len, d_model)
        K:          torch.Tensor Key tensor of shape (batch, seq_len, d_model)
        V:          torch.Tensor Value tensor of shape (batch, seq_len, d_model)
        is_causal:  bool = False Whether to apply causal masking
        """
        
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        assert Q.dtype and K.dtype and V.dtype in supported_dtypes, \
            f"Unsupported dtype for Q: {Q.dtype}, K: {K.dtype}, V: {V.dtype}"
        assert Q.dtype == K.dtype == V.dtype, \
            f"All tensors must have same dtype. Got Q: {Q.dtype}, K: {K.dtype}, V: {V.dtype}"
        assert Q.ndim == K.ndim == V.ndim == 3, \
            f"All tensors must be 3D. Got Q: {Q.ndim}D, K: {K.ndim}D, V: {V.ndim}D"
        assert (
            Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        ), "Our pointer arithmetic will assume contiguous Q, K, V"
        
        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape
        ctx.is_causal = is_causal

        # initialize O, L
        O = torch.zeros_like(Q)
        L = torch.zeros((B, N_QUERIES), device=Q.device, dtype=Q.dtype)

        scale = 1.0 / math.sqrt(D)
        
        # launch kernel
        grid = lambda meta: (triton.cdiv(N_QUERIES, meta['Q_TILE_SIZE']), B)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            is_causal=ctx.is_causal
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute flash attention backward pass.
        
        grad_output:    torch.Tensor Gradient w.r.t. output of shape (batch, seq_len, d_model)
        """
        
        Q, K, V, O, L = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        assert grad_output.is_cuda, f"grad_output must be on CUDA. Got: {grad_output.device}"
        assert grad_output.is_contiguous(), f"grad_output must be contiguous. Got strides: {grad_output.stride()}"
        assert grad_output.shape == O.shape, f"grad_output shape mismatch. Expected: {O.shape}, got: {grad_output.shape}"
        assert grad_output.dtype == Q.dtype, f"grad_output dtype mismatch. Expected: {Q.dtype}, got: {grad_output.dtype}"
        
        if torch.is_grad_enabled():
            assert torch.isfinite(grad_output).all(), "grad_output contains non-finite values"

        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        scale = 1.0 / math.sqrt(D)
        D_ = (grad_output * O).sum(dim=-1)
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dO = grad_output
        
        # launch kernel
        grid_dkv = lambda meta: (triton.cdiv(N_KEYS, meta['K_TILE_SIZE']), B)
        flash_backward_dkv_kernel[grid_dkv](
            Q, K, V,
            L, D_,
            dK, dV, dO,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D_.stride(0), D_.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            is_causal=ctx.is_causal
        )
        grid_dq = lambda meta: (triton.cdiv(N_QUERIES, meta['Q_TILE_SIZE']), B)
        flash_backward_dq_kernel[grid_dq](
            Q, K, V,
            L, D_,
            dQ, dO,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D_.stride(0), D_.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            is_causal=ctx.is_causal
        )
        
        return dQ.to(Q.dtype), dK, dV, None
