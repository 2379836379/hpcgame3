import torch

from ref import reference_sparse_attention
from solution import sparse_attention


def _build_index(B, H_K, N, top_k, bs, M, device):
    max_blocks = (M + bs - 1) // bs
    index = torch.empty((B, H_K, N, top_k), device=device, dtype=torch.int32)
    for i in range(N):
        block0 = min(i // bs, max_blocks - 1)
        block1 = max_blocks - 1
        index[:, :, i, 0] = block0
        index[:, :, i, 1] = block1
    return index


def run():
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, H_Q, H_K, N, M, D_H, TOP_K, BS = 1, 2, 1, 12, 10, 16, 2, 4
    sm_scale = 1.0 / (D_H**0.5)

    q = torch.randn((B, H_Q, N, D_H), device=device, dtype=torch.float16)
    k = torch.randn((B, H_K, M, D_H), device=device, dtype=torch.float16)
    v = torch.randn((B, H_K, M, D_H), device=device, dtype=torch.float16)
    index = _build_index(B, H_K, N, TOP_K, BS, M, device)

    out_ref = reference_sparse_attention(q, k, v, index, BS, sm_scale)
    out_sol = sparse_attention(q, k, v, index, BS, sm_scale)

    torch.testing.assert_close(out_sol, out_ref, atol=1e-2, rtol=1e-2)
    print("PASS")


if __name__ == "__main__":
    run()
