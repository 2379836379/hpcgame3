#include <vector>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <omp.h>
void calculate(uint32_t n, uint32_t m, uint32_t *edges, uint64_t *dis)
{
    const int T = 16;
    omp_set_dynamic(0);
    omp_set_num_threads(T);
    
    const uint64_t INF = std::numeric_limits<uint64_t>::max();
    
    // ==================== 1. 并行构建CSR ====================
    std::vector<uint32_t> head(n + 1, 0);
    std::vector<uint32_t> deg(n, 0);
    
    #pragma omp parallel
    {
        std::vector<uint32_t> local_deg(n, 0);
        #pragma omp for nowait schedule(static)
        for (uint32_t i = 0; i < m; ++i) {
            ++local_deg[edges[i * 3]];
        }
        #pragma omp critical
        {
            for (uint32_t i = 0; i < n; ++i) deg[i] += local_deg[i];
        }
    }
    
    for (uint32_t i = 0; i < n; ++i) {
        head[i + 1] = head[i] + deg[i];
    }
    
    std::vector<uint32_t> to(m), wgt(m);
    std::vector<uint32_t> cur = head;
    
    for (uint32_t i = 0; i < m; ++i) {
        uint32_t u = edges[i * 3];
        uint32_t v = edges[i * 3 + 1];
        uint32_t w = edges[i * 3 + 2];
        uint32_t pos = cur[u]++;
        to[pos] = v;
        wgt[pos] = w;
    }
    
    // ==================== 2. 计算Delta参数 ====================
    uint64_t max_wgt = 1;
    #pragma omp parallel for reduction(max:max_wgt) schedule(static)
    for (uint32_t i = 0; i < m; ++i) {
        uint64_t w = edges[i * 3 + 2];
        if (w > max_wgt) max_wgt = w;
    }
    
    double avg_deg = (m > 0) ? (double)m / n : 1.0;
    uint64_t delta = std::max((uint64_t)1, max_wgt / std::max((uint64_t)avg_deg, (uint64_t)1));
    
    // ==================== 3. 初始化距离数组 ====================
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n; ++i) {
        dis[i] = INF;
    }
    dis[0] = 0;
    
    // ==================== 4. Delta-Stepping ====================
    size_t num_buckets = (max_wgt * n / delta) + 100;
    num_buckets = std::min(num_buckets, (size_t)10000000);
    
    std::vector<std::vector<uint32_t>> buckets(num_buckets);
    std::vector<uint8_t> in_bucket(n, 0);
    
    std::vector<std::vector<std::pair<uint32_t, uint64_t>>> thread_buf(T);
    for (int t = 0; t < T; ++t) {
        thread_buf[t].reserve(4096);
    }
    
    buckets[0].push_back(0);
    in_bucket[0] = 1;
    
    size_t cur_bucket = 0;
    
    while (cur_bucket < num_buckets) {
        // 找下一个非空桶
        while (cur_bucket < num_buckets && buckets[cur_bucket].empty()) {
            ++cur_bucket;
        }
        if (cur_bucket >= num_buckets) break;
        
        // 处理当前桶的轻边（权重 <= delta），可能多轮
        while (!buckets[cur_bucket].empty()) {
            std::vector<uint32_t> frontier;
            frontier.swap(buckets[cur_bucket]);
            
            for (uint32_t u : frontier) {
                in_bucket[u] = 0;
            }
            
            for (int t = 0; t < T; ++t) {
                thread_buf[t].clear();
            }
            
            // 并行松弛轻边
            #pragma omp parallel num_threads(T)
            {
                int tid = omp_get_thread_num();
                auto &buf = thread_buf[tid];
                
                #pragma omp for schedule(dynamic, 64)
                for (size_t i = 0; i < frontier.size(); ++i) {
                    uint32_t u = frontier[i];
                    uint64_t du = dis[u];
                    if (du == INF) continue;
                    
                    for (uint32_t ei = head[u]; ei < head[u + 1]; ++ei) {
                        uint32_t v = to[ei];
                        uint32_t w = wgt[ei];
                        
                        if (w > delta) continue; // 只处理轻边
                        
                        uint64_t nd = du + (uint64_t)w;
                        uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                        
                        while (nd < old) {
                            if (__atomic_compare_exchange_n(&dis[v], &old, nd, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                                buf.emplace_back(v, nd);
                                break;
                            }
                        }
                    }
                }
            }
            
            // 收集结果入桶
            for (int t = 0; t < T; ++t) {
                for (auto &p : thread_buf[t]) {
                    uint32_t v = p.first;
                    uint64_t d = p.second;
                    
                    // 检查距离是否仍然有效
                    if (dis[v] != d) continue;
                    
                    size_t b = d / delta;
                    if (b >= num_buckets) b = num_buckets - 1;
                    
                    if (!in_bucket[v]) {
                        in_bucket[v] = 1;
                        buckets[b].push_back(v);
                    }
                }
            }
        }
        
        // 处理重边（权重 > delta）
        // 收集当前桶范围内所有已确定的节点
        std::vector<uint32_t> heavy_frontier;
        uint64_t bucket_min = cur_bucket * delta;
        uint64_t bucket_max = (cur_bucket + 1) * delta;
        
        for (uint32_t i = 0; i < n; ++i) {
            uint64_t d = dis[i];
            if (d >= bucket_min && d < bucket_max) {
                heavy_frontier.push_back(i);
            }
        }
        
        if (!heavy_frontier.empty()) {
            for (int t = 0; t < T; ++t) {
                thread_buf[t].clear();
            }
            
            #pragma omp parallel num_threads(T)
            {
                int tid = omp_get_thread_num();
                auto &buf = thread_buf[tid];
                
                #pragma omp for schedule(dynamic, 64)
                for (size_t i = 0; i < heavy_frontier.size(); ++i) {
                    uint32_t u = heavy_frontier[i];
                    uint64_t du = dis[u];
                    
                    for (uint32_t ei = head[u]; ei < head[u + 1]; ++ei) {
                        uint32_t v = to[ei];
                        uint32_t w = wgt[ei];
                        
                        if (w <= delta) continue; // 只处理重边
                        
                        uint64_t nd = du + (uint64_t)w;
                        uint64_t old = __atomic_load_n(&dis[v], __ATOMIC_RELAXED);
                        
                        while (nd < old) {
                            if (__atomic_compare_exchange_n(&dis[v], &old, nd, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                                buf.emplace_back(v, nd);
                                break;
                            }
                        }
                    }
                }
            }
            
            // 收集重边松弛结果入桶
            for (int t = 0; t < T; ++t) {
                for (auto &p : thread_buf[t]) {
                    uint32_t v = p.first;
                    uint64_t d = p.second;
                    
                    if (dis[v] != d) continue;
                    
                    size_t b = d / delta;
                    if (b >= num_buckets) b = num_buckets - 1;
                    
                    if (!in_bucket[v]) {
                        in_bucket[v] = 1;
                        buckets[b].push_back(v);
                    }
                }
            }
        }
        
        ++cur_bucket;
    }
}