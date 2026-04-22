#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <omp.h>
struct PushItem {
    uint64_t b;
    uint32_t v;
};
void calculate(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
{
    omp_set_dynamic(0);
    omp_set_num_threads(16);
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
    // Heuristic delta from max weight (can be tuned)
    uint64_t max_w = 1;
    for (uint32_t i = 0; i < m; ++i) max_w = std::max<uint64_t>(max_w, wgt[i]);
    uint64_t delta = std::max<uint64_t>(1, (uint64_t)std::sqrt((double)max_w));
    auto bucket_index = [&](uint64_t d) -> uint64_t { return d / delta; };
    // Assume caller pre-inits dis[] to "INF"; only force source=0.
    if (__atomic_load_n(&dis[0], __ATOMIC_RELAXED) != 0ULL)
        __atomic_store_n(&dis[0], 0ULL, __ATOMIC_RELAXED);
    // buckets + min-heap of non-empty bucket indices (no per-vertex in_bucket dedup; allow duplicates)
    std::vector<std::vector<uint32_t>> buckets(1);
    std::vector<uint64_t> heap; // min-heap via push_heap/pop_heap with greater<>
    auto heap_push = [&](uint64_t bi) {
        heap.push_back(bi);
        std::push_heap(heap.begin(), heap.end(), std::greater<uint64_t>());
    };
    auto heap_pop = [&]() -> uint64_t {
        std::pop_heap(heap.begin(), heap.end(), std::greater<uint64_t>());
        uint64_t bi = heap.back();
        heap.pop_back();
        return bi;
    };
    auto ensure_bucket = [&](uint64_t bi) {
        if (bi >= buckets.size()) buckets.resize((size_t)bi + 1);
    };
    auto push_to_bucket_single = [&](uint32_t v, uint64_t dist) {
        uint64_t bi = bucket_index(dist);
        ensure_bucket(bi);
        bool was_empty = buckets[bi].empty();
        buckets[bi].push_back(v);
        if (was_empty) heap_push(bi);
    };
    push_to_bucket_single(0, 0);
    const int T = 16;
    std::vector<std::vector<uint32_t>> add_cur(T);
    std::vector<std::vector<PushItem>> add_future(T);
    std::vector<size_t> offsets(T, 0);
    std::vector<uint32_t> R, S, R_next;
    uint64_t cur_bucket = 0;
    int done = 0;
    int cont_light = 0;
    // One persistent parallel region to reduce OpenMP overhead
    #pragma omp parallel num_threads(16) shared(done, cont_light, cur_bucket, R, S, R_next, buckets, heap, offsets, add_cur, add_future, dis)
    {
        int tid = omp_get_thread_num();
        while (true) {
            // Pick next non-empty bucket
            #pragma omp single
            {
                done = 0;
                while (!heap.empty()) {
                    uint64_t bi = heap_pop();
                    if (bi < buckets.size() && !buckets[bi].empty()) {
                        cur_bucket = bi;
                        R.swap(buckets[bi]);
                        S.clear();
                        break;
                    }
                }
                if (R.empty()) done = 1;
            }
            #pragma omp barrier
            if (done) break;
            // Filter stale entries (node might be here with newer distance in another bucket)
            #pragma omp single
            {
                size_t w = 0;
                for (size_t i = 0; i < R.size(); ++i) {
                    uint32_t u = R[i];
                    uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                    if (bucket_index(du) == cur_bucket) R[w++] = u;
                }
                R.resize(w);
            }
            #pragma omp barrier
            // Light-edge closure
            while (true) {
                #pragma omp single
                cont_light = (R.empty() ? 0 : 1);
                #pragma omp barrier
                if (!cont_light) break;
                add_cur[tid].clear();
                add_future[tid].clear();
                #pragma omp for schedule(static)
                for (size_t idx = 0; idx < R.size(); ++idx) {
                    uint32_t u = R[idx];
                    uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                    for (uint32_t ei = head[u], eend = head[u + 1]; ei < eend; ++ei) {
                        uint32_t w = wgt[ei];
                        if ((uint64_t)w >= delta) continue; // light only
                        uint32_t v = to[ei];
                        uint64_t nd = du + (uint64_t)w;
                        uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                        while (nd < old) {
                            if (__atomic_compare_exchange_n(&dis[v], &old, nd, false,
                                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                                uint64_t bi = bucket_index(nd);
                                if (bi == cur_bucket) add_cur[tid].push_back(v);
                                else add_future[tid].push_back(PushItem{bi, v});
                                break;
                            }
                            // old updated, retry if still nd < old
                        }
                    }
                }
                // Merge "future bucket" pushes and build R_next from add_cur (prefix-sum + parallel copy)
                #pragma omp barrier
                #pragma omp single
                {
                    // merge future pushes (single thread for vector safety)
                    uint64_t max_b = 0;
                    for (int t = 0; t < T; ++t)
                        for (const auto &it : add_future[t])
                            if (it.b > max_b) max_b = it.b;
                    ensure_bucket(max_b);
                    for (int t = 0; t < T; ++t) {
                        for (const auto &it : add_future[t]) {
                            bool was_empty = buckets[it.b].empty();
                            buckets[it.b].push_back(it.v);
                            if (was_empty) heap_push(it.b);
                        }
                    }
                    // prefix sum for R_next
                    size_t total = 0;
                    for (int t = 0; t < T; ++t) {
                        offsets[t] = total;
                        total += add_cur[t].size();
                    }
                    R_next.assign(total, 0);
                }
                #pragma omp barrier
                // parallel copy add_cur -> R_next
                #pragma omp for schedule(static)
                for (int t = 0; t < T; ++t) {
                    size_t off = offsets[t];
                    auto &vec = add_cur[t];
                    for (size_t i = 0; i < vec.size(); ++i)
                        R_next[off + i] = vec[i];
                }
                #pragma omp barrier
                #pragma omp single
                {
                    // S += R; R = R_next
                    S.insert(S.end(), R.begin(), R.end());
                    R.swap(R_next);
                }
                #pragma omp barrier
            }
            // Heavy edges from S
            add_future[tid].clear();
            #pragma omp for schedule(static)
            for (size_t idx = 0; idx < S.size(); ++idx) {
                uint32_t u = S[idx];
                uint64_t du = __atomic_load_n(&dis[u], __ATOMIC_RELAXED);
                for (uint32_t ei = head[u], eend = head[u + 1]; ei < eend; ++ei) {
                    uint32_t w = wgt[ei];
                    if ((uint64_t)w < delta) continue; // heavy only
                    uint32_t v = to[ei];
                    uint64_t nd = du + (uint64_t)w;
                    uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                    while (nd < old) {
                        if (__atomic_compare_exchange_n(&dis[v], &old, nd, false,
                                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            add_future[tid].push_back(PushItem{bucket_index(nd), v});
                            break;
                        }
                    }
                }
            }
            // Merge heavy pushes into buckets
            #pragma omp barrier
            #pragma omp single
            {
                uint64_t max_b = 0;
                for (int t = 0; t < T; ++t)
                    for (const auto &it : add_future[t])
                        if (it.b > max_b) max_b = it.b;
                ensure_bucket(max_b);
                for (int t = 0; t < T; ++t) {
                    for (const auto &it : add_future[t]) {
                        bool was_empty = buckets[it.b].empty();
                        buckets[it.b].push_back(it.v);
                        if (was_empty) heap_push(it.b);
                    }
                }
                R.clear();
            }
            #pragma omp barrier
        }
    }
}