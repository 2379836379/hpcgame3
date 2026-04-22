#include <vector>
#include <cstdint>
#include <cmath>
#include <utility>
#include <queue>
#include <limits>
#include <functional>
#include <algorithm>
#include <omp.h>
// 并行 Delta-stepping 实现，优化桶管理与局部缓冲以减少同步开销
void calculate(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
{
    const int THREADS = 16;
    omp_set_num_threads(THREADS);
    std::vector<uint32_t> outdeg(n, 0);
    for (uint32_t i = 0; i < m; ++i)
        ++outdeg[edges[i * 3]];
    std::vector<uint32_t> head(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i)
        head[i + 1] = head[i] + outdeg[i];
    std::vector<uint32_t> to(m);
    std::vector<uint32_t> wgt(m);
    std::vector<uint32_t> cursor = head;
    for (uint32_t i = 0; i < m; ++i)
    {
        uint32_t u = edges[i * 3];
        uint32_t pos = cursor[u]++;
        to[pos] = edges[i * 3 + 1];
        wgt[pos] = edges[i * 3 + 2];
    }
    uint64_t max_w = 1;
    for (uint32_t i = 0; i < m; ++i)
        max_w = std::max<uint64_t>(max_w, wgt[i]);
    uint64_t delta = std::max<uint64_t>(1, static_cast<uint64_t>(std::sqrt(static_cast<double>(max_w))));
    std::vector<std::vector<uint32_t>> buckets(1);
    std::vector<uint8_t> bucket_ready(1, 0);
    using BucketQueue = std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>;
    BucketQueue ready;
    auto ensure_bucket = [&](uint64_t idx)
    {
        if (idx >= buckets.size())
        {
            buckets.resize(idx + 1);
            bucket_ready.resize(idx + 1, 0);
        }
    };
    auto push_bucket = [&](uint32_t node, uint64_t dist)
    {
        uint64_t idx = dist / delta;
        ensure_bucket(idx);
        buckets[idx].push_back(node);
        if (!bucket_ready[idx])
        {
            bucket_ready[idx] = 1;
            ready.push(idx);
        }
    };
    if (__atomic_load_n(&dis[0], __ATOMIC_RELAXED) != 0ULL)
        __atomic_store_n(&dis[0], 0ULL, __ATOMIC_RELAXED);
    push_bucket(0, 0);
    int thread_count = omp_get_max_threads();
    std::vector<std::vector<uint32_t>> light_buffer(thread_count);
    std::vector<std::vector<std::pair<uint64_t, uint32_t>>> heavy_buffer(thread_count);
    const uint64_t INF64 = std::numeric_limits<uint64_t>::max();
    while (true)
    {
        uint64_t cur_bucket = std::numeric_limits<uint64_t>::max();
        while (!ready.empty())
        {
            uint64_t idx = ready.top();
            ready.pop();
            if (idx < buckets.size() && !buckets[idx].empty())
            {
                cur_bucket = idx;
                break;
            }
            if (idx < bucket_ready.size())
                bucket_ready[idx] = 0;
        }
        if (cur_bucket == std::numeric_limits<uint64_t>::max())
            break;
        std::vector<uint32_t> R;
        R.swap(buckets[cur_bucket]);
        bucket_ready[cur_bucket] = 0;
        std::vector<uint32_t> active;
        active.reserve(R.size());
        for (uint32_t u : R)
        {
            uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
            if (du != INF64 && du / delta == cur_bucket)
                active.push_back(u);
        }
        if (active.empty())
            continue;
        std::vector<uint32_t> S;
        S.reserve(active.size());
        R.swap(active);
        while (!R.empty())
        {
            for (auto &vec : light_buffer)
                vec.clear();
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto &my_add = light_buffer[tid];
                #pragma omp for schedule(dynamic)
                for (size_t idx = 0; idx < R.size(); ++idx)
                {
                    uint32_t u = R[idx];
                    uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                    for (uint32_t ei = head[u]; ei < head[u + 1]; ++ei)
                    {
                        uint32_t v = to[ei];
                        uint32_t w = wgt[ei];
                        if (w >= delta)
                            continue;
                        uint64_t nd = du + static_cast<uint64_t>(w);
                        uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                        while (nd < old)
                        {
                            if (__atomic_compare_exchange_n(&dis[v], &old, nd, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                            {
                                my_add.push_back(v);
                                break;
                            }
                        }
                    }
                }
            }
            size_t total = 0;
            for (auto &vec : light_buffer)
                total += vec.size();
            std::vector<uint32_t> next;
            next.reserve(total);
            for (auto &vec : light_buffer)
                next.insert(next.end(), vec.begin(), vec.end());
            S.insert(S.end(), R.begin(), R.end());
            R.swap(next);
        }
        for (auto &vec : heavy_buffer)
            vec.clear();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &my_push = heavy_buffer[tid];
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
                        continue;
                    uint64_t nd = du + static_cast<uint64_t>(w);
                    uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                    while (nd < old)
                    {
                        if (__atomic_compare_exchange_n(&dis[v], &old, nd, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                        {
                            my_push.emplace_back(nd, v);
                            break;
                        }
                    }
                }
            }
        }
        for (auto &vec : heavy_buffer)
            for (auto &entry : vec)
                push_bucket(entry.second, entry.first);
    }
}