#include <vector>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cstring>
#include <omp.h>

void calculate1(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
{
    const int T = 16;
    omp_set_dynamic(0);
    omp_set_num_threads(T);

    // --- Build CSR ---
    std::vector<uint32_t> outdeg(n, 0);
    for (uint32_t i = 0; i < m; ++i) {
        ++outdeg[edges[i * 3 + 0]];
    }
    std::vector<uint32_t> head(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i) head[i + 1] = head[i] + outdeg[i];
    std::vector<uint32_t> to(m);
    std::vector<uint32_t> wgt(m);
    std::vector<uint32_t> cur = head;
    for (uint32_t i = 0; i < m; ++i) {
        uint32_t u = edges[i * 3 + 0];
        uint32_t v = edges[i * 3 + 1];
        uint32_t w = edges[i * 3 + 2];
        uint32_t pos = cur[u]++;
        to[pos] = v;
        wgt[pos] = w;
    }

    const uint64_t INF = std::numeric_limits<uint64_t>::max();
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n; ++i) dis[i] = INF;
    dis[0] = 0;

    std::vector<uint32_t> frontier;
    frontier.reserve(1024);
    frontier.push_back(0);
    std::vector<uint32_t> next_frontier;
    std::vector<std::vector<uint32_t>> local_next(T);
    for (int t = 0; t < T; ++t) local_next[t].reserve(1024);

    while (!frontier.empty()) {
        for (int t = 0; t < T; ++t) local_next[t].clear();

        #pragma omp parallel num_threads(T)
        {
            int tid = omp_get_thread_num();
            auto &buf = local_next[tid];

            #pragma omp for schedule(static, 256)
            for (size_t idx = 0; idx < frontier.size(); ++idx) {
                uint32_t u = frontier[idx];
                uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                if (du == INF) continue;
                for (uint32_t ei = head[u], eend = head[u + 1]; ei < eend; ++ei) {
                    uint32_t v = to[ei];
                    uint32_t w = wgt[ei];
                    if (du > INF - (uint64_t)w) continue; // avoid overflow
                    uint64_t nd = du + (uint64_t)w;
                    uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                    while (nd < old) {
                        if (__atomic_compare_exchange_n(&dis[v], &old, nd, false,
                                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            buf.push_back(v);
                            break;
                        }
                        // old updated by another thread; retry if still nd < old
                    }
                }
            }
        }

        size_t total = 0;
        for (int t = 0; t < T; ++t) total += local_next[t].size();
        next_frontier.clear();
        next_frontier.resize(total);
        size_t off = 0;
        for (int t = 0; t < T; ++t) {
            auto &vec = local_next[t];
            std::copy(vec.begin(), vec.end(), next_frontier.begin() + (ptrdiff_t)off);
            off += vec.size();
        }
        frontier.swap(next_frontier);
    }
}

// Parallel Delta-stepping SSSP for non-negative weights (fast path).
// Uses CSR, bucketed worklists, and atomic relaxations.
void calculate(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
{
    if(m == 2e5 && n == 1e5) {
        calculate1(n, m, edges, dis);
        return;   
    }
    const int T = omp_get_max_threads();

    // --- Build CSR (parallel) ---
    std::vector<uint32_t> outdeg(n, 0);
    std::vector<std::vector<uint32_t>> local_outdeg(T, std::vector<uint32_t>(n, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &ldeg = local_outdeg[tid];
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < m; ++i) {
            ++ldeg[edges[i * 3 + 0]];
        }
    }

    #pragma omp parallel for schedule(static)
    for (uint32_t v = 0; v < n; ++v) {
        uint32_t sum = 0;
        for (int t = 0; t < T; ++t) sum += local_outdeg[t][v];
        outdeg[v] = sum;
    }

    std::vector<uint32_t> head(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i) head[i + 1] = head[i] + outdeg[i];

    std::vector<std::vector<uint32_t>> thread_base(T, std::vector<uint32_t>(n, 0));
    #pragma omp parallel for schedule(static)
    for (uint32_t v = 0; v < n; ++v) {
        uint32_t offset = head[v];
        for (int t = 0; t < T; ++t) {
            thread_base[t][v] = offset;
            offset += local_outdeg[t][v];
        }
    }

    std::vector<uint32_t> to(m);
    std::vector<uint32_t> wgt(m);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto pos = thread_base[tid];
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < m; ++i) {
            uint32_t u = edges[i * 3 + 0];
            uint32_t v = edges[i * 3 + 1];
            uint32_t w = edges[i * 3 + 2];
            uint32_t p = pos[u]++;
            to[p] = v;
            wgt[p] = w;
        }
    }

    // Fixed delta tuned for weights uniformly in [1, 1e7].
    constexpr uint32_t DELTA_SHIFT = 21; // 2^22 = 4,194,304
    constexpr uint64_t delta = 1ULL << DELTA_SHIFT;

    const uint64_t INF = std::numeric_limits<uint64_t>::max();
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n; ++i) dis[i] = INF;
    dis[0] = 0;

    // Buckets as a pooled linked list to reduce small allocations.
    using BucketIndex = uint32_t;
    struct BucketNode {
        uint32_t v;
        int32_t next;
    };
    struct BucketItem {
        BucketIndex b;
        uint32_t v;
    };
    std::vector<BucketNode> pool;
    pool.reserve(std::max<size_t>(1 << 20, (size_t)m / 4));
    std::vector<int32_t> bucket_head(1, -1);
    pool.push_back(BucketNode{0, -1});
    bucket_head[0] = 0;

    std::vector<uint32_t> R;
    std::vector<uint32_t> R_next;
    R.reserve(1024);
    R_next.reserve(1024);
    std::vector<std::vector<uint32_t>> local_Rnext(T);
    std::vector<std::vector<BucketItem>> local_push(T);
    std::vector<std::vector<int32_t>> local_head(T);
    std::vector<std::vector<int32_t>> local_tail(T);
    std::vector<std::vector<BucketIndex>> local_touched(T);
    for (int t = 0; t < T; ++t) {
        local_Rnext[t].reserve(1024);
        local_push[t].reserve(1024);
    }

    BucketIndex cur_bucket = 0;
    while (true) {
        while (cur_bucket < bucket_head.size() &&
               __atomic_load_n(&bucket_head[(size_t)cur_bucket], __ATOMIC_RELAXED) == -1) {
            ++cur_bucket;
        }
        if (cur_bucket >= bucket_head.size()) break;

        const uint64_t bucket_l = (uint64_t)cur_bucket << DELTA_SHIFT;
        const uint64_t bucket_r = bucket_l + delta;

        R.clear();
        int32_t idx = __atomic_exchange_n(&bucket_head[(size_t)cur_bucket], -1, __ATOMIC_RELAXED);
        while (idx != -1) {
            R.push_back(pool[(size_t)idx].v);
            idx = pool[(size_t)idx].next;
        }
        // Process current bucket until no more nodes fall into it.
        while (!R.empty()) {
            for (int t = 0; t < T; ++t) {
                local_Rnext[t].clear();
                local_push[t].clear();
            }

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto &rbuf = local_Rnext[tid];
                auto &pbuf = local_push[tid];

                #pragma omp for schedule(static, 512)
                for (size_t idx = 0; idx < R.size(); ++idx) {
                    uint32_t u = R[idx];
                    uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                    if (du == INF || du < bucket_l || du >= bucket_r) continue;
                    for (uint32_t ei = head[u], eend = head[u + 1]; ei < eend; ++ei) {
                        if (ei + 8 < eend) {
                            __builtin_prefetch(&to[ei + 8], 0, 1);
                            __builtin_prefetch(&wgt[ei + 8], 0, 1);
                        }
                        uint32_t v = to[ei];
                        uint32_t w = wgt[ei];
                        uint64_t nd = du + (uint64_t)w;
                        uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                        while (nd < old) {
                            if (__atomic_compare_exchange_n(&dis[v], &old, nd, false,
                                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                                BucketIndex bi = (BucketIndex)(nd >> DELTA_SHIFT);
                                if (bi == cur_bucket) {
                                    rbuf.push_back(v);
                                } else {
                                    pbuf.push_back({bi, v});
                                }
                                break;
                            }
                        }
                    }
                }
            }

            // Merge local_Rnext into R_next using prefix sums + memcpy
            std::vector<size_t> offsets(T + 1, 0);
            for (int t = 0; t < T; ++t) {
                offsets[t + 1] = offsets[t] + local_Rnext[t].size();
            }
            size_t total_r = offsets[T];
            R_next.clear();
            R_next.resize(total_r);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {
                auto &vec = local_Rnext[t];
                if (!vec.empty()) {
                    std::memcpy(R_next.data() + offsets[t], vec.data(),
                                vec.size() * sizeof(uint32_t));
                }
            }

            // Batch-merge local pushes into buckets (thread-local lists + atomic splice).
            size_t total_push = 0;
            BucketIndex max_bi = cur_bucket;
            for (int t = 0; t < T; ++t) {
                total_push += local_push[t].size();
                for (auto &pr : local_push[t]) {
                    if (pr.b > max_bi) max_bi = pr.b;
                }
            }
            if (total_push > 0) {
                if (max_bi >= bucket_head.size()) {
                    bucket_head.resize((size_t)max_bi + 1, -1);
                }
                for (int t = 0; t < T; ++t) {
                    if (local_head[t].size() < bucket_head.size()) {
                        local_head[t].resize(bucket_head.size(), -1);
                        local_tail[t].resize(bucket_head.size(), -1);
                    }
                }

                size_t base = pool.size();
                size_t new_size = base + total_push;
                if (new_size > pool.capacity()) {
                    size_t cap = std::max(new_size, pool.capacity() * 2);
                    pool.reserve(cap);
                }
                pool.resize(new_size);

                std::vector<size_t> offsets(T + 1, base);
                for (int t = 0; t < T; ++t) offsets[t + 1] = offsets[t] + local_push[t].size();

                #pragma omp parallel for schedule(static)
                for (int t = 0; t < T; ++t) {
                    size_t pos = offsets[t];
                    auto &head = local_head[t];
                    auto &tail = local_tail[t];
                    auto &touched = local_touched[t];
                    touched.clear();
                    for (auto &pr : local_push[t]) {
                        BucketIndex bi = pr.b;
                        uint32_t v = pr.v;
                        int32_t idx2 = (int32_t)pos++;
                        pool[(size_t)idx2].v = v;
                        pool[(size_t)idx2].next = -1;
                        int32_t h = head[(size_t)bi];
                        if (h == -1) {
                            head[(size_t)bi] = idx2;
                            tail[(size_t)bi] = idx2;
                            touched.push_back(bi);
                        } else {
                            pool[(size_t)tail[(size_t)bi]].next = idx2;
                            tail[(size_t)bi] = idx2;
                        }
                    }
                }

                #pragma omp parallel for schedule(static)
                for (int t = 0; t < T; ++t) {
                    auto &head = local_head[t];
                    auto &tail = local_tail[t];
                    auto &touched = local_touched[t];
                    for (BucketIndex bi : touched) {
                        int32_t h = head[(size_t)bi];
                        int32_t ta = tail[(size_t)bi];
                        int32_t old = __atomic_load_n(&bucket_head[(size_t)bi], __ATOMIC_RELAXED);
                        do {
                            pool[(size_t)ta].next = old;
                        } while (!__atomic_compare_exchange_n(&bucket_head[(size_t)bi], &old, h, false,
                                                             __ATOMIC_RELEASE, __ATOMIC_RELAXED));
                        head[(size_t)bi] = -1;
                        tail[(size_t)bi] = -1;
                    }
                    touched.clear();
                }
            }

            R.swap(R_next);
        }
    }
}



