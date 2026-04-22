#include <vector>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <omp.h>

// Parallel label-correcting SSSP for non-negative weights.
// Uses a worklist (frontier) with atomic relaxations; safe for duplicate pushes.
void calculate(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
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
