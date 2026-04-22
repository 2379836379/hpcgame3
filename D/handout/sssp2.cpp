#include <vector>
#include <cstdint>
#include <cmath>
#include <utility>
#include <omp.h>

// 并行 Delta-stepping 实现，针对非负权重的 SSSP 问题。
// 该实现构建邻接表，然后使用宽度为 delta 的桶（bucket）并行处理轻边和重边。
// 选择 delta = max(1, (uint64_t)sqrt(MAX_WEIGHT)) 作为经验值。

// 不依赖外部常量：在函数内部根据输入边权计算 max_weight，并使用本地 delta

void calculate(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
{
    omp_set_num_threads(16);

    // 构建邻接表（CSR 格式）以提升遍历的缓存局部性
    std::vector<uint32_t> outdeg(n, 0);
    for (uint32_t i = 0; i < m; ++i)
    {
        uint32_t u = edges[i * 3];
        outdeg[u]++;
    }

    // head 大小为 n+1，边索引范围为 [head[u], head[u+1])
    std::vector<uint32_t> head(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i)
        head[i + 1] = head[i] + outdeg[i];

    std::vector<uint32_t> to(m);
    std::vector<uint32_t> wgt(m);
    std::vector<uint32_t> ptr = head; // 填充游标
    for (uint32_t i = 0; i < m; ++i)
    {
        uint32_t u = edges[i * 3];
        uint32_t v = edges[i * 3 + 1];
        uint32_t w = edges[i * 3 + 2];
        uint32_t pos = ptr[u]++;
        to[pos] = v;
        wgt[pos] = w;
    }

    // 选择 delta：根据输入边权计算最大权重，避免依赖外部常量
    uint64_t max_w = 1;
    for (uint32_t i = 0; i < m; ++i)
        if (wgt[i] > max_w)
            max_w = wgt[i];
    uint64_t delta = static_cast<uint64_t>(std::max<uint64_t>(1, static_cast<uint64_t>(std::sqrt((double)max_w))));

    // buckets：动态扩展的向量，每个桶保存当前待处理的节点
    // 同时维护每个桶的 in_bucket 标记数组以便去重（in_bucket[b][v]==1 表示 v 已在该桶中）
    std::vector<std::vector<uint32_t>> buckets;
    // 使用 vector<bool> 进行位打包，每个节点 1 bit，提高缓存利用率并降低内存带宽
    std::vector<std::vector<bool>> in_bucket; // per-bucket presence flags (bit-packed)
    buckets.emplace_back();
    in_bucket.emplace_back(n, 0);

    auto bucket_index = [&](uint64_t d) -> uint64_t { return d / delta; };

    // 将源点放入第 0 桶（假设调用方已将 dis[0]=0）
    if (dis[0] != 0)
        __atomic_store_n(&dis[0], 0ULL, __ATOMIC_RELAXED);
    in_bucket[0][0] = true;
    buckets[0].push_back(0);

    uint64_t cur = 0;
    while (true)
    {
        // 找到下一个非空桶
        while (cur < buckets.size() && buckets[cur].empty())
            ++cur;
        if (cur >= buckets.size())
            break;

        // R 初始化为当前桶的所有节点；同时清空该桶的 in_bucket 标记
        std::vector<uint32_t> R;
        R.swap(buckets[cur]);
        for (uint32_t u : R)
            in_bucket[cur][u] = false;

        // S 收集本桶将被处理完的所有节点（包括在轻边松弛中加入的节点）
        std::vector<uint32_t> S;

        // 处理轻边（weight < delta），需要对 R 做闭包直到无新加入
        while (!R.empty())
        {
            // 将 R 的节点并行松弛其轻边，产生新的节点加入 R_next
            std::vector<std::vector<uint32_t>> local_add(omp_get_max_threads());

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                std::vector<uint32_t> &my_add = local_add[tid];

                #pragma omp for schedule(dynamic)
                for (size_t idx = 0; idx < R.size(); ++idx)
                {
                    uint32_t u = R[idx];
                    uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                    // 遍历 CSR 中 u 的出边范围 [head[u], head[u+1])
                    for (uint32_t ei = head[u]; ei < head[u + 1]; ++ei)
                    {
                        uint32_t v = to[ei];
                        uint32_t w = wgt[ei];
                        if (w >= delta)
                            continue; // 仅处理轻边
                        uint64_t nd = du + (uint64_t)w;
                        uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                        while (nd < old)
                        {
                            if (__atomic_compare_exchange_n(&dis[v], &old, nd, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                            {
                                my_add.push_back(v);
                                break;
                            }
                            // old 已更新，继续判断
                        }
                    }
                }
            }

            // 合并 local_add 到 R_next（并去重会更复杂，这里允许重复，后续松弛会被忽略）
            std::vector<uint32_t> R_next;
            for (auto &vec : local_add)
            {
                if (!vec.empty())
                {
                    R_next.insert(R_next.end(), vec.begin(), vec.end());
                }
            }

            // 将当前 R 的节点加入 S（即最终将在本桶处理的集合）
            S.insert(S.end(), R.begin(), R.end());

            // R = R_next，继续处理轻边的闭包
            R.swap(R_next);
        }

        // 处理 S 中节点的重边（weight >= delta），将被松弛的节点放入对应桶
        // 并行处理 S 中节点的重边：改为线程本地缓冲，避免每次成功松弛都进入 critical
        int T = omp_get_max_threads();
        std::vector<std::vector<std::pair<uint64_t, uint32_t>>> local_push(T);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(dynamic)
            for (size_t idx = 0; idx < S.size(); ++idx)
            {
                uint32_t u = S[idx];
                uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                for (uint32_t ei = head[u]; ei < head[u + 1]; ++ei)
                {
                    uint32_t v = to[ei];
                    uint32_t w = wgt[ei];
                    if (w < delta)
                        continue; // 仅处理重边
                    uint64_t nd = du + (uint64_t)w;
                    uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                    while (nd < old)
                    {
                        if (__atomic_compare_exchange_n(&dis[v], &old, nd, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                        {
                            uint64_t bi = bucket_index(nd);
                            // 先加入线程本地缓冲，稍后再统一合并到全局 buckets
                            local_push[tid].emplace_back(bi, v);
                            break;
                        }
                        // old 已更新，继续判断
                    }
                }
            }
        }

        // 并行区外单线程批量合并各线程缓冲到全局 buckets（保证扩容与 push 的串行性）
        for (int t = 0; t < T; ++t)
        {
            for (auto &pr : local_push[t])
            {
                size_t bi = static_cast<size_t>(pr.first);
                uint32_t vv = pr.second;
                // 确保 buckets 与 in_bucket 足够大
                while (bi >= buckets.size())
                {
                    buckets.emplace_back();
                    in_bucket.emplace_back(n, 0);
                }
                // 去重：如果该节点尚未在目标桶中则入桶并标记
                if (!in_bucket[bi][vv])
                {
                    in_bucket[bi][vv] = true;
                    buckets[bi].push_back(vv);
                }
            }
            local_push[t].clear();
        }

        // 完成当前桶，继续查找下一个非空桶
    }
}