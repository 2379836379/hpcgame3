#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <utility>
#include <limits>
#include <omp.h>

// Parallel label-correcting SSSP for non-negative weights.
// Uses a worklist (frontier) with atomic relaxations; safe for duplicate pushes.
void calculate(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
{
    const int T = omp_get_max_threads();

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
    std::vector<uint8_t> in_queue(n, 0);
    in_queue[0] = 1;
    std::vector<uint32_t> next_frontier;
    std::vector<std::vector<uint32_t>> local_next(T);
    for (int t = 0; t < T; ++t) local_next[t].reserve(1024);

    while (!frontier.empty()) {
        for (int t = 0; t < T; ++t) local_next[t].clear();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &buf = local_next[tid];
            std::vector<std::pair<uint32_t, uint64_t>> updates;
            updates.reserve(1024);

            #pragma omp for schedule(static, 256)
            for (size_t idx = 0; idx < frontier.size(); ++idx) {
                uint32_t u = frontier[idx];
                __atomic_store_n(&in_queue[u], 0, __ATOMIC_RELAXED);
                uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                if (du == INF) continue;
                for (uint32_t ei = head[u], eend = head[u + 1]; ei < eend; ++ei) {
                    uint32_t v = to[ei];
                    uint32_t w = wgt[ei];
                    if (du > INF - (uint64_t)w) continue; // avoid overflow
                    uint64_t nd = du + (uint64_t)w;
                    uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                    if (nd < old) {
                        updates.emplace_back(v, nd);
                    }
                }
            }

            if (!updates.empty()) {
                std::sort(updates.begin(), updates.end(),
                          [](const auto &a, const auto &b) { return a.first < b.first; });
                size_t wpos = 0;
                for (size_t i = 1; i < updates.size(); ++i) {
                    if (updates[i].first == updates[wpos].first) {
                        if (updates[i].second < updates[wpos].second) {
                            updates[wpos].second = updates[i].second;
                        }
                    } else {
                        ++wpos;
                        updates[wpos] = updates[i];
                    }
                }
                updates.resize(wpos + 1);

                for (auto &up : updates) {
                    uint32_t v = up.first;
                    uint64_t nd = up.second;
                    uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                    while (nd < old) {
                        if (__atomic_compare_exchange_n(&dis[v], &old, nd, false,
                                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            if (__atomic_exchange_n(&in_queue[v], 1, __ATOMIC_RELAXED) == 0) {
                                buf.push_back(v);
                            }
                            break;
                        }
                        // old updated; retry if still nd < old
                    }
                }
            }
        }

        size_t total = 0;
        std::vector<size_t> offsets(T + 1, 0);
        for (int t = 0; t < T; ++t) {
            offsets[t + 1] = offsets[t] + local_next[t].size();
        }
        total = offsets[T];
        next_frontier.clear();
        next_frontier.resize(total);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) {
            auto &vec = local_next[t];
            if (!vec.empty()) {
                std::memcpy(next_frontier.data() + offsets[t], vec.data(),
                            vec.size() * sizeof(uint32_t));
            }
        }
        frontier.swap(next_frontier);
    }
}
