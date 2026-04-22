import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - fallback for CPU-only envs
    triton = None
    tl = None


def _reference_sparse_attention(q, k, v, index, bs, sm_scale):
    B, h_q, N, d_h = q.shape
    _, h_k, M, _ = k.shape
    _, _, _, top_k = index.shape
    g = h_q // h_k
    device = q.device

    out = torch.zeros_like(q)

    k_f = k.float()
    v_f = v.float()
    q_f = q.float()

    for hk_idx in range(h_k):
        k_group = k_f[:, hk_idx, :, :]
        v_group = v_f[:, hk_idx, :, :]

        q_group = q_f[:, hk_idx * g : (hk_idx + 1) * g, :, :]
        idx_group = index[:, hk_idx, :, :]

        chunk_size = 128
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            cur_N = i_end - i_start

            q_chunk = q_group[:, :, i_start:i_end, :]
            idx_chunk = idx_group[:, i_start:i_end, :]

            offsets = torch.arange(bs, device=device)
            t_idx = (idx_chunk.unsqueeze(-1) * bs + offsets).reshape(
                B, cur_N, top_k * bs
            )

            mask_invalid = t_idx < 0
            t_idx_clamped = t_idx.clamp(min=0, max=M - 1)

            b_idx = torch.arange(B, device=device).view(B, 1, 1)
            k_selected = k_group[b_idx, t_idx_clamped]
            v_selected = v_group[b_idx, t_idx_clamped]

            scores = torch.matmul(
                q_chunk.unsqueeze(3), k_selected.unsqueeze(1).transpose(-1, -2)
            ).squeeze(3)
            scores = scores * sm_scale

            i_global = torch.arange(i_start, i_end, device=device).view(1, 1, cur_N, 1)
            mask_causal = t_idx.unsqueeze(1) > i_global

            scores.masked_fill_(mask_invalid.unsqueeze(1) | mask_causal, float("-inf"))

            probs = torch.softmax(scores, dim=-1).unsqueeze(3)
            chunk_out = torch.matmul(probs, v_selected.unsqueeze(1)).squeeze(3)

            out[:, hk_idx * g : (hk_idx + 1) * g, i_start:i_end, :] = chunk_out.to(
                out.dtype
            )

    return out


if triton is not None:

    @triton.jit
    def _sparse_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        idx_ptr,
        out_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ib,
        stride_ih,
        stride_in,
        stride_it,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        B,
        H_Q,
        H_K,
        N,
        M,
        D_H,
        sm_scale,
        g,
        TOP_K: tl.constexpr,
        BLOCK_BS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid % N
        tmp = pid // N
        hq = tmp % H_Q
        b = tmp // H_Q

        d_offsets = tl.arange(0, BLOCK_D)
        q_ptrs = (
            q_ptr
            + b * stride_qb
            + hq * stride_qh
            + n * stride_qn
            + d_offsets * stride_qd
        )
        q = tl.load(q_ptrs, mask=d_offsets < D_H, other=0).to(tl.float32)

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        m_i = 0.0
        l_i = 0.0
        has_valid = tl.zeros((), dtype=tl.int1)

        hk = hq // g
        t_offsets = tl.arange(0, TOP_K)
        idx_ptrs = (
            idx_ptr
            + b * stride_ib
            + hk * stride_ih
            + n * stride_in
            + t_offsets * stride_it
        )
        block_ids = tl.load(idx_ptrs)

        k_base = k_ptr + b * stride_kb + hk * stride_kh
        v_base = v_ptr + b * stride_vb + hk * stride_vh

        offs = tl.arange(0, BLOCK_BS)
        neg_inf = tl.full((), -1e9, tl.float32)

        for t in range(TOP_K):
            block_id = block_ids[t]
            token_idx = block_id * BLOCK_BS + offs
            valid = (block_id >= 0) & (token_idx <= n)

            token_idx_load = tl.where(block_id >= 0, token_idx, 0)
            token_idx_load = tl.minimum(token_idx_load, M - 1)

            k_ptrs = (
                k_base + token_idx_load[:, None] * stride_kn + d_offsets[None] * stride_kd
            )
            v_ptrs = (
                v_base + token_idx_load[:, None] * stride_vn + d_offsets[None] * stride_vd
            )
            k = tl.load(
                k_ptrs,
                mask=valid[:, None] & (d_offsets[None] < D_H),
                other=0,
            ).to(tl.float32)
            v = tl.load(
                v_ptrs,
                mask=valid[:, None] & (d_offsets[None] < D_H),
                other=0,
            ).to(tl.float32)

            scores = tl.sum(k * q[None, :], axis=1) * sm_scale
            scores = tl.where(valid, scores, neg_inf)

            block_max = tl.max(scores, axis=0)
            exp_scores = tl.where(valid, tl.exp(scores - block_max), 0.0)
            exp_sum = tl.sum(exp_scores, axis=0)
            weighted_v = tl.sum(exp_scores[:, None] * v, axis=0)

            block_valid = exp_sum > 0
            m_j = tl.maximum(m_i, block_max)
            exp_m = tl.exp(m_i - m_j)
            l1 = l_i * exp_m + exp_sum
            acc1 = acc * exp_m + weighted_v

            m0 = block_max
            l0 = exp_sum
            acc0 = weighted_v

            use0 = (has_valid == 0) & block_valid
            use1 = (has_valid != 0) & block_valid

            m_i = tl.where(use1, m_j, tl.where(use0, m0, m_i))
            l_i = tl.where(use1, l1, tl.where(use0, l0, l_i))
            acc = tl.where(use1, acc1, tl.where(use0, acc0, acc))
            has_valid = has_valid | block_valid

        l_i_safe = tl.where(l_i == 0, 1.0, l_i)
        out = acc / l_i_safe

        out_ptrs = (
            out_ptr
            + b * stride_ob
            + hq * stride_oh
            + n * stride_on
            + d_offsets * stride_od
        )
        tl.store(out_ptrs, out, mask=d_offsets < D_H)


def sparse_attention(q, k, v, index, bs, sm_scale):
    if (
        triton is None
        or not q.is_cuda
        or not k.is_cuda
        or not v.is_cuda
        or not index.is_cuda
    ):
        return _reference_sparse_attention(q, k, v, index, bs, sm_scale)

    B, H_Q, N, D_H = q.shape
    _, H_K, M, _ = k.shape
    top_k = index.shape[-1]
    g = H_Q // H_K

    if D_H > 128:
        return _reference_sparse_attention(q, k, v, index, bs, sm_scale)

    if D_H <= 32:
        block_d = 32
    elif D_H <= 64:
        block_d = 64
    else:
        block_d = 128

    out = torch.empty_like(q)

    grid = (B * H_Q * N,)
    _sparse_attention_kernel[grid](
        q,
        k,
        v,
        index,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H_Q,
        H_K,
        N,
        M,
        D_H,
        sm_scale,
        g,
        TOP_K=top_k,
        BLOCK_BS=bs,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    return out
