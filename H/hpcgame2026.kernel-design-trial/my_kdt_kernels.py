from typing import Dict

import kdt
import torch

def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 1:
        return vector_add_kernel
    if task_id == 2:
        return matmul_kernel
    if task_id == 3:
        return MatmulFineScaleDispatcher()
    if task_id == 4:
        return flash_attention_kernel
    raise ValueError(f"Unsupported task_id: {task_id}")



def calculate_num_jobs1(task_args: dict[str, int]) -> int:
    # 单个 job 处理整个向量
    return 1
@kdt.kernel(num_jobs_calculator=calculate_num_jobs1)
def vector_add_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    # 单个 job 处理全量数据；双缓冲 + 就地输出，尽量重叠 load 与 VXM 计算
    # 4 个 tile * 8192 * 4B = 128KB，正好填满 SPM
    BLOCK_SIZE = 8192
    vec_size = task_args['N']
    #assert vec_size % BLOCK_SIZE == 0, "vec_size 必须是 BLOCK_SIZE 的整数倍"
    num_blocks = vec_size // BLOCK_SIZE

    # 分配 SPM 上的数据块（双缓冲）
    a0 = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    b0 = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    a1 = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    b1 = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')

    # 预取第 0 块
    kdt.load(io_tensors['a'][0:BLOCK_SIZE], a0)
    kdt.load(io_tensors['b'][0:BLOCK_SIZE], b0)

    for i in range(num_blocks):
        start_idx = i * BLOCK_SIZE
        end_idx = start_idx + BLOCK_SIZE

        if i % 2 == 0:
            # 计算当前块
            kdt.add(a0, b0, a0)
            # 预取下一块到另一组 buffer（尽量与计算重叠）
            if i + 1 < num_blocks:
                next_start = end_idx
                next_end = next_start + BLOCK_SIZE
                kdt.load(io_tensors['a'][next_start:next_end], a1)
                kdt.load(io_tensors['b'][next_start:next_end], b1)

            # 写回结果
            kdt.store(a0, io_tensors['c'][start_idx:end_idx])
        else:
            kdt.add(a1, b1, a1)
            if i + 1 < num_blocks:
                next_start = end_idx
                next_end = next_start + BLOCK_SIZE
                kdt.load(io_tensors['a'][next_start:next_end], a0)
                kdt.load(io_tensors['b'][next_start:next_end], b0)
            kdt.store(a1, io_tensors['c'][start_idx:end_idx])





def calculate_num_jobs_matmul(task_args: dict[str, int]) -> int:
    TILE_M = 128
    TILE_N = 128
    m_tiles = task_args['M'] // TILE_M
    n_tiles = task_args['N'] // TILE_N
    return m_tiles * n_tiles
@kdt.kernel(num_jobs_calculator=calculate_num_jobs_matmul)
def matmul_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    # 单个 job 负责一个 (TILE_M x TILE_N) 的输出块，K 维度分块累加
    TILE_M = 128
    TILE_N = 128
    TILE_K = 128

    n = task_args['N']
    k = task_args['K']

    tiles_n = n // TILE_N
    job_id = kdt.get_job_id()
    tile_m = job_id // tiles_n
    tile_n = job_id % tiles_n

    m_start = tile_m * TILE_M
    m_end = m_start + TILE_M
    n_start = tile_n * TILE_N
    n_end = n_start + TILE_N

    # 双缓冲 A/B，C 在 SPM 内累加
    a0 = kdt.alloc_spm((TILE_M, TILE_K), dtype='float32')
    b0 = kdt.alloc_spm((TILE_K, TILE_N), dtype='float32')
    a1 = kdt.alloc_spm((TILE_M, TILE_K), dtype='float32')
    b1 = kdt.alloc_spm((TILE_K, TILE_N), dtype='float32')
    c_tile = kdt.alloc_spm((TILE_M, TILE_N), dtype='float32')
    kdt.fill(c_tile, 0.0)

    num_k_blocks = k // TILE_K

    # 预取第 0 个 K 块
    kdt.load(io_tensors['A'][m_start:m_end, 0:TILE_K], a0)
    kdt.load(io_tensors['B'][0:TILE_K, n_start:n_end], b0)

    for kk in range(num_k_blocks):
        k_start = kk * TILE_K
        k_end = k_start + TILE_K

        if kk % 2 == 0:
            kdt.matmul(a0, b0, c_tile, accumulate=True)
            if kk + 1 < num_k_blocks:
                next_k_start = k_end
                next_k_end = next_k_start + TILE_K
                kdt.load(io_tensors['A'][m_start:m_end, next_k_start:next_k_end], a1)
                kdt.load(io_tensors['B'][next_k_start:next_k_end, n_start:n_end], b1)
        else:
            kdt.matmul(a1, b1, c_tile, accumulate=True)
            if kk + 1 < num_k_blocks:
                next_k_start = k_end
                next_k_end = next_k_start + TILE_K
                kdt.load(io_tensors['A'][m_start:m_end, next_k_start:next_k_end], a0)
                kdt.load(io_tensors['B'][next_k_start:next_k_end, n_start:n_end], b0)

    kdt.store(c_tile, io_tensors['C'][m_start:m_end, n_start:n_end])


def calculate_num_jobs_matmul_fine128(task_args: dict[str, int]) -> int:
    TILE_M = 128
    TILE_N = 128
    m_tiles = task_args['M'] // TILE_M
    n_tiles = task_args['N'] // TILE_N
    return m_tiles * n_tiles
@kdt.kernel(num_jobs_calculator=calculate_num_jobs_matmul_fine128)
def matmul_fine_scale_kernel_128(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    # 任务3（M=512 专用）：细粒度缩放矩阵乘法
    # A = Ab * As（K 维每 64 个元素共享一个 scale）
    # B = Bb * Bs（K 维每 64 行共享一个 scale）
    TILE_M = 128
    TILE_N = 128
    TILE_K = 128
    SCALE_GRANULARITY = 64
    n = task_args['N']
    k = task_args['K']
    k_scale = k // SCALE_GRANULARITY
    tiles_n = n // TILE_N
    job_id = kdt.get_job_id()
    tile_m = job_id // tiles_n
    tile_n = job_id % tiles_n
    # 每个 job 负责一个 128x128 输出块（tile_m, tile_n）
    m_start = tile_m * TILE_M
    m_end = m_start + TILE_M
    n_start = tile_n * TILE_N
    n_end = n_start + TILE_N
    # 输出块坐标范围：[m_start:m_end, n_start:n_end]
    a0 = kdt.alloc_spm((TILE_M, TILE_K), dtype='float32')
    b0 = kdt.alloc_spm((TILE_K, TILE_N), dtype='float32')
    a1 = kdt.alloc_spm((TILE_M, TILE_K), dtype='float32')
    b1 = kdt.alloc_spm((TILE_K, TILE_N), dtype='float32')
    a2 = kdt.alloc_spm((TILE_M, TILE_K), dtype='float32')
    b2 = kdt.alloc_spm((TILE_K, TILE_N), dtype='float32')
    # 三缓存：a0/b0, a1/b1, a2/b2 用于 load/scale/matmul 流水
    as_full = kdt.alloc_spm((TILE_M, k_scale), dtype='float32')
    bs_full = kdt.alloc_spm((k_scale, TILE_N), dtype='float32')
    # As/Bs 全量加载进 SPM，避免每个 K-block 反复 load
    c_tile = kdt.alloc_spm((TILE_M, TILE_N), dtype='float32')
    # c_tile 在 SPM 内累加所有 K-block 的乘积
    num_k_blocks = k // TILE_K
    # K 维分块数量（向上取整，末尾不足 TILE_K 的块会做零填充）

    kdt.load(io_tensors['As'][m_start:m_end, 0:k_scale], as_full)
    kdt.load(io_tensors['Bs'][0:k_scale, n_start:n_end], bs_full)
    # 预先加载当前输出块对应的 As/Bs

    # 预取第 0/1 块
    prefetch0_k_start = 0
    prefetch0_k_end = prefetch0_k_start + TILE_K
    if prefetch0_k_end > k:
        prefetch0_k_end = k
    prefetch0_k_len = prefetch0_k_end - prefetch0_k_start
    kdt.load(io_tensors['Ab'][m_start:m_end, prefetch0_k_start:prefetch0_k_end], a0[:, 0:prefetch0_k_len])
    kdt.load(io_tensors['Bb'][prefetch0_k_start:prefetch0_k_end, n_start:n_end], b0[0:prefetch0_k_len, :])

    if num_k_blocks > 1:
        prefetch1_k_start = TILE_K
        prefetch1_k_end = prefetch1_k_start + TILE_K
        if prefetch1_k_end > k:
            prefetch1_k_end = k
        prefetch1_k_len = prefetch1_k_end - prefetch1_k_start
        kdt.load(io_tensors['Ab'][m_start:m_end, prefetch1_k_start:prefetch1_k_end], a1[:, 0:prefetch1_k_len])
        kdt.load(io_tensors['Bb'][prefetch1_k_start:prefetch1_k_end, n_start:n_end], b1[0:prefetch1_k_len, :])

    # 先缩放第 0 块，保证流水首轮能直接进入 matmul
    scale0_k_start = 0
    scale0_k_end = scale0_k_start + TILE_K
    if scale0_k_end > k:
        scale0_k_end = k
    scale0_k_len = scale0_k_end - scale0_k_start
    for s in range(scale0_k_len // SCALE_GRANULARITY):
        sg = SCALE_GRANULARITY
        a_sub = a0[:, s * sg:(s + 1) * sg]
        a_scale = as_full[:, s:s + 1]
        a_scale_b = kdt.broadcast_to(a_scale, 1, sg)
        kdt.mul(a_sub, a_scale_b, a_sub)
        b_sub = b0[s * sg:(s + 1) * sg, :]
        b_scale = bs_full[s:s + 1, :]
        b_scale_b = kdt.broadcast_to(b_scale, 0, sg)
        kdt.mul(b_sub, b_scale_b, b_sub)

    # 三阶段固定顺序流水：load(next) → scale(curr) → matmul(prev)
    for kk in range(num_k_blocks):
        if kk % 3 == 0:
                        # matmul prev（kk）在 a0/b0 上进行
            kdt.matmul(a0, b0, c_tile, accumulate=True)
            # load next（kk+2）到 a2/b2
            if kk + 2 < num_k_blocks:
                next_k_start = (kk + 2) * TILE_K
                next_k_end = next_k_start + TILE_K
                if next_k_end > k:
                    next_k_end = k
                next_k_len = next_k_end - next_k_start
                kdt.load(io_tensors['Ab'][m_start:m_end, next_k_start:next_k_end], a2[:, 0:next_k_len])
                kdt.load(io_tensors['Bb'][next_k_start:next_k_end, n_start:n_end], b2[0:next_k_len, :])

            # scale curr（kk+1）在 a1/b1 上进行
            if kk + 1 < num_k_blocks:
                k_start = (kk + 1) * TILE_K
                k_end = k_start + TILE_K
                if k_end > k:
                    k_end = k
                k_len = k_end - k_start
                scale_base = k_start // SCALE_GRANULARITY  # As/Bs 中对应的 scale 起始列/行
                for s in range(k_len // SCALE_GRANULARITY):
                    sg = SCALE_GRANULARITY
                    a_sub = a1[:, s * sg:(s + 1) * sg]
                    a_scale = as_full[:, scale_base + s:scale_base + s + 1]
                    a_scale_b = kdt.broadcast_to(a_scale, 1, sg)
                    kdt.mul(a_sub, a_scale_b, a_sub)
                    b_sub = b1[s * sg:(s + 1) * sg, :]
                    b_scale = bs_full[scale_base + s:scale_base + s + 1, :]
                    b_scale_b = kdt.broadcast_to(b_scale, 0, sg)
                    kdt.mul(b_sub, b_scale_b, b_sub)



        elif kk % 3 == 1:
            kdt.matmul(a1, b1, c_tile, accumulate=True)
            # load next（kk+2）到 a0/b0
            if kk + 2 < num_k_blocks:
                next_k_start = (kk + 2) * TILE_K
                next_k_end = next_k_start + TILE_K
                if next_k_end > k:
                    next_k_end = k
                next_k_len = next_k_end - next_k_start
                kdt.load(io_tensors['Ab'][m_start:m_end, next_k_start:next_k_end], a0[:, 0:next_k_len])
                kdt.load(io_tensors['Bb'][next_k_start:next_k_end, n_start:n_end], b0[0:next_k_len, :])

            # scale curr（kk+1）在 a2/b2 上进行
            if kk + 1 < num_k_blocks:
                k_start = (kk + 1) * TILE_K
                k_end = k_start + TILE_K
                if k_end > k:
                    k_end = k
                k_len = k_end - k_start
                scale_base = k_start // SCALE_GRANULARITY  # As/Bs 中对应的 scale 起始列/行
                for s in range(k_len // SCALE_GRANULARITY):
                    sg = SCALE_GRANULARITY
                    a_sub = a2[:, s * sg:(s + 1) * sg]
                    a_scale = as_full[:, scale_base + s:scale_base + s + 1]
                    a_scale_b = kdt.broadcast_to(a_scale, 1, sg)
                    kdt.mul(a_sub, a_scale_b, a_sub)
                    b_sub = b2[s * sg:(s + 1) * sg, :]
                    b_scale = bs_full[scale_base + s:scale_base + s + 1, :]
                    b_scale_b = kdt.broadcast_to(b_scale, 0, sg)
                    kdt.mul(b_sub, b_scale_b, b_sub)

            # matmul prev（kk）在 a1/b1 上进行
            

        else:
            kdt.matmul(a2, b2, c_tile, accumulate=True)
            # load next（kk+2）到 a1/b1
            if kk + 2 < num_k_blocks:
                next_k_start = (kk + 2) * TILE_K
                next_k_end = next_k_start + TILE_K
                if next_k_end > k:
                    next_k_end = k
                next_k_len = next_k_end - next_k_start
                kdt.load(io_tensors['Ab'][m_start:m_end, next_k_start:next_k_end], a1[:, 0:next_k_len])
                kdt.load(io_tensors['Bb'][next_k_start:next_k_end, n_start:n_end], b1[0:next_k_len, :])

            # scale curr（kk+1）在 a0/b0 上进行
            if kk + 1 < num_k_blocks:
                k_start = (kk + 1) * TILE_K
                k_end = k_start + TILE_K
                if k_end > k:
                    k_end = k
                k_len = k_end - k_start
                scale_base = k_start // SCALE_GRANULARITY  # As/Bs 中对应的 scale 起始列/行
                for s in range(k_len // SCALE_GRANULARITY):
                    sg = SCALE_GRANULARITY
                    a_sub = a0[:, s * sg:(s + 1) * sg]
                    a_scale = as_full[:, scale_base + s:scale_base + s + 1]
                    a_scale_b = kdt.broadcast_to(a_scale, 1, sg)
                    kdt.mul(a_sub, a_scale_b, a_sub)
                    b_sub = b0[s * sg:(s + 1) * sg, :]
                    b_scale = bs_full[scale_base + s:scale_base + s + 1, :]
                    b_scale_b = kdt.broadcast_to(b_scale, 0, sg)
                    kdt.mul(b_sub, b_scale_b, b_sub)

            # matmul prev（kk）在 a2/b2 上进行
            
    kdt.store(c_tile, io_tensors['C'][m_start:m_end, n_start:n_end])


def calculate_num_jobs_matmul_fine_256(task_args: dict[str, int]) -> int:
    TILE_M = 256
    TILE_N = 128
    m_tiles = task_args['M'] // TILE_M
    n_tiles = task_args['N'] // TILE_N
    return m_tiles * n_tiles


@kdt.kernel(num_jobs_calculator=calculate_num_jobs_matmul_fine_256)
def matmul_fine_scale_kernel_256(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    TILE_M = 256
    TILE_N = 128
    TILE_K = 128
    SCALE_GRANULARITY = 64

    n = task_args['N']
    k = task_args['K']

    tiles_n = n // TILE_N
    job_id = kdt.get_job_id()
    tile_m = job_id // tiles_n
    tile_n = job_id % tiles_n

    m_start = tile_m * TILE_M
    m_end = m_start + TILE_M
    n_start = tile_n * TILE_N
    n_end = n_start + TILE_N

    a0 = kdt.alloc_spm((TILE_M, TILE_K), dtype='float32')
    b0 = kdt.alloc_spm((TILE_K, TILE_N), dtype='float32')
    a1 = kdt.alloc_spm((TILE_M, TILE_K), dtype='float32')
    b1 = kdt.alloc_spm((TILE_K, TILE_N), dtype='float32')

    as0 = kdt.alloc_spm((TILE_M, TILE_K // SCALE_GRANULARITY), dtype='float32')
    bs0 = kdt.alloc_spm((TILE_K // SCALE_GRANULARITY, TILE_N), dtype='float32')
    as1 = kdt.alloc_spm((TILE_M, TILE_K // SCALE_GRANULARITY), dtype='float32')
    bs1 = kdt.alloc_spm((TILE_K // SCALE_GRANULARITY, TILE_N), dtype='float32')

    c_tile = kdt.alloc_spm((TILE_M, TILE_N), dtype='float32')
    kdt.fill(c_tile, 0.0)

    num_k_blocks = k // TILE_K

    kdt.load(io_tensors['Ab'][m_start:m_end, 0:TILE_K], a0)
    kdt.load(io_tensors['Bb'][0:TILE_K, n_start:n_end], b0)
    kdt.load(io_tensors['As'][m_start:m_end, 0:(TILE_K // SCALE_GRANULARITY)], as0)
    kdt.load(io_tensors['Bs'][0:(TILE_K // SCALE_GRANULARITY), n_start:n_end], bs0)

    for kk in range(num_k_blocks):
        k_start = kk * TILE_K
        k_end = k_start + TILE_K

        if kk % 2 == 0:
            for s in range(TILE_K // SCALE_GRANULARITY):
                a_sub = a0[:, s * SCALE_GRANULARITY:(s + 1) * SCALE_GRANULARITY]
                a_scale = as0[:, s:s + 1]
                a_scale_b = kdt.broadcast_to(a_scale, 1, SCALE_GRANULARITY)
                kdt.mul(a_sub, a_scale_b, a_sub)

                b_sub = b0[s * SCALE_GRANULARITY:(s + 1) * SCALE_GRANULARITY, :]
                b_scale = bs0[s:s + 1, :]
                b_scale_b = kdt.broadcast_to(b_scale, 0, SCALE_GRANULARITY)
                kdt.mul(b_sub, b_scale_b, b_sub)

            kdt.matmul(a0, b0, c_tile, accumulate=True)

            if kk + 1 < num_k_blocks:
                next_k_start = k_end
                next_k_end = next_k_start + TILE_K
                kdt.load(io_tensors['Ab'][m_start:m_end, next_k_start:next_k_end], a1)
                kdt.load(io_tensors['Bb'][next_k_start:next_k_end, n_start:n_end], b1)
                kdt.load(io_tensors['As'][m_start:m_end, (next_k_start // SCALE_GRANULARITY):(next_k_end // SCALE_GRANULARITY)], as1)
                kdt.load(io_tensors['Bs'][(next_k_start // SCALE_GRANULARITY):(next_k_end // SCALE_GRANULARITY), n_start:n_end], bs1)
        else:
            for s in range(TILE_K // SCALE_GRANULARITY):
                a_sub = a1[:, s * SCALE_GRANULARITY:(s + 1) * SCALE_GRANULARITY]
                a_scale = as1[:, s:s + 1]
                a_scale_b = kdt.broadcast_to(a_scale, 1, SCALE_GRANULARITY)
                kdt.mul(a_sub, a_scale_b, a_sub)

                b_sub = b1[s * SCALE_GRANULARITY:(s + 1) * SCALE_GRANULARITY, :]
                b_scale = bs1[s:s + 1, :]
                b_scale_b = kdt.broadcast_to(b_scale, 0, SCALE_GRANULARITY)
                kdt.mul(b_sub, b_scale_b, b_sub)

            kdt.matmul(a1, b1, c_tile, accumulate=True)

            if kk + 1 < num_k_blocks:
                next_k_start = k_end
                next_k_end = next_k_start + TILE_K
                kdt.load(io_tensors['Ab'][m_start:m_end, next_k_start:next_k_end], a0)
                kdt.load(io_tensors['Bb'][next_k_start:next_k_end, n_start:n_end], b0)
                kdt.load(io_tensors['As'][m_start:m_end, (next_k_start // SCALE_GRANULARITY):(next_k_end // SCALE_GRANULARITY)], as0)
                kdt.load(io_tensors['Bs'][(next_k_start // SCALE_GRANULARITY):(next_k_end // SCALE_GRANULARITY), n_start:n_end], bs0)

    kdt.store(c_tile, io_tensors['C'][m_start:m_end, n_start:n_end])


class MatmulFineScaleDispatcher:
    def compile(self, task_args: Dict[str, int], io_tensors: Dict[str, torch.Tensor]):
        if task_args['M'] == 512:
            return matmul_fine_scale_kernel_128.compile(task_args, io_tensors)
        return matmul_fine_scale_kernel_256.compile(task_args, io_tensors)
   

def calculate_num_jobs_flash_attention(task_args: dict[str, int]) -> int:
    BLOCK_Q = 128
    return task_args['S_qo'] // BLOCK_Q

@kdt.kernel(num_jobs_calculator=calculate_num_jobs_flash_attention)
def flash_attention_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BLOCK_Q = 128
    BLOCK_K = 128
    D = 128
    EXP_BASE = 2.7182818284590451

    s_kv = task_args['S_kv']
    num_k_blocks = s_kv // BLOCK_K

    job_id = kdt.get_job_id()
    q_start = job_id * BLOCK_Q
    q_end = q_start + BLOCK_Q

    q_tile = kdt.alloc_spm((BLOCK_Q, D), dtype='float32')
    k0 = kdt.alloc_spm((BLOCK_K, D), dtype='float32')
    v0 = kdt.alloc_spm((BLOCK_K, D), dtype='float32')
    k1 = kdt.alloc_spm((BLOCK_K, D), dtype='float32')
    v1 = kdt.alloc_spm((BLOCK_K, D), dtype='float32')

    score0 = kdt.alloc_spm((BLOCK_Q, BLOCK_K), dtype='float32')
    score1 = kdt.alloc_spm((BLOCK_Q, BLOCK_K), dtype='float32')
    o_tile = kdt.alloc_spm((BLOCK_Q, D), dtype='float32')

    m = kdt.alloc_spm((BLOCK_Q,), dtype='float32')
    l = kdt.alloc_spm((BLOCK_Q,), dtype='float32')
    row_max = kdt.alloc_spm((BLOCK_Q,), dtype='float32')
    sum_p = kdt.alloc_spm((BLOCK_Q,), dtype='float32')
    m_scale = kdt.alloc_spm((BLOCK_Q,), dtype='float32')
    m_next = kdt.alloc_spm((BLOCK_Q,), dtype='float32')

    kdt.load(io_tensors['Q'][q_start:q_end, 0:D], q_tile)
    kdt.fill(o_tile, 0.0)
    kdt.fill(m, -1.0e9)
    kdt.fill(l, 0.0)

    kdt.load(io_tensors['K'][0:BLOCK_K, 0:D], k0)
    kdt.load(io_tensors['V'][0:BLOCK_K, 0:D], v0)
    if num_k_blocks > 1:
        kdt.load(io_tensors['K'][BLOCK_K:2 * BLOCK_K, 0:D], k1)
        kdt.load(io_tensors['V'][BLOCK_K:2 * BLOCK_K, 0:D], v1)

    cur_k_t = kdt.transpose(k0, 0, 1)
    kdt.matmul(q_tile, cur_k_t, score0, accumulate=False)

    for kk in range(num_k_blocks):
        next_k_start = (kk + 2) * BLOCK_K
        next_k_end = next_k_start + BLOCK_K
        if kk % 2 == 0:
            if kk + 1 < num_k_blocks:
                next_k_t = kdt.transpose(k1, 0, 1)
                kdt.matmul(q_tile, next_k_t, score1, accumulate=False)

            kdt.reduce(score0, 1, 'max', row_max)
            kdt.max(m, row_max, m_next)
            kdt.sub(m, m_next, m_scale)
            kdt.exp(m_scale, m_scale, EXP_BASE)

            row_max_u = kdt.unsqueeze(row_max, 1)
            row_max_b = kdt.broadcast_to(row_max_u, 1, BLOCK_K)
            kdt.sub(score0, row_max_b, score0)
            kdt.exp(score0, score0, EXP_BASE)
            kdt.sub(row_max, m_next, row_max)
            kdt.exp(row_max, row_max, EXP_BASE)
            row_max_u2 = kdt.unsqueeze(row_max, 1)
            row_max_b2 = kdt.broadcast_to(row_max_u2, 1, BLOCK_K)
            kdt.mul(score0, row_max_b2, score0)

            kdt.reduce(score0, 1, 'sum', sum_p)

            m_scale_u = kdt.unsqueeze(m_scale, 1)
            m_scale_b = kdt.broadcast_to(m_scale_u, 1, D)
            kdt.mul(o_tile, m_scale_b, o_tile)
            kdt.matmul(score0, v0, o_tile, accumulate=True)

            kdt.mul(l, m_scale, l)
            kdt.add(l, sum_p, l)
            kdt.copy(m_next, m)

            if kk + 2 < num_k_blocks:
                kdt.load(io_tensors['K'][next_k_start:next_k_end, 0:D], k0)
                kdt.load(io_tensors['V'][next_k_start:next_k_end, 0:D], v0)
        else:
            if kk + 1 < num_k_blocks:
                next_k_t = kdt.transpose(k0, 0, 1)
                kdt.matmul(q_tile, next_k_t, score0, accumulate=False)

            kdt.reduce(score1, 1, 'max', row_max)
            kdt.max(m, row_max, m_next)
            kdt.sub(m, m_next, m_scale)
            kdt.exp(m_scale, m_scale, EXP_BASE)

            row_max_u = kdt.unsqueeze(row_max, 1)
            row_max_b = kdt.broadcast_to(row_max_u, 1, BLOCK_K)
            kdt.sub(score1, row_max_b, score1)
            kdt.exp(score1, score1, EXP_BASE)
            kdt.sub(row_max, m_next, row_max)
            kdt.exp(row_max, row_max, EXP_BASE)
            row_max_u2 = kdt.unsqueeze(row_max, 1)
            row_max_b2 = kdt.broadcast_to(row_max_u2, 1, BLOCK_K)
            kdt.mul(score1, row_max_b2, score1)

            kdt.reduce(score1, 1, 'sum', sum_p)

            m_scale_u = kdt.unsqueeze(m_scale, 1)
            m_scale_b = kdt.broadcast_to(m_scale_u, 1, D)
            kdt.mul(o_tile, m_scale_b, o_tile)
            kdt.matmul(score1, v1, o_tile, accumulate=True)

            kdt.mul(l, m_scale, l)
            kdt.add(l, sum_p, l)
            kdt.copy(m_next, m)

            if kk + 2 < num_k_blocks:
                kdt.load(io_tensors['K'][next_k_start:next_k_end, 0:D], k1)
                kdt.load(io_tensors['V'][next_k_start:next_k_end, 0:D], v1)

    l_u = kdt.unsqueeze(l, 1)
    l_b = kdt.broadcast_to(l_u, 1, D)
    kdt.div(o_tile, l_b, o_tile)
    kdt.store(o_tile, io_tensors['O'][q_start:q_end, 0:D])
