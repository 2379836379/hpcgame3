#include "spgemm_topk.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

static inline int owner_rank(int key, int size) {
  int r = key % size;
  return (r < 0) ? (r + size) : r;
}

static inline uint64_t make_key(int r, int c) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(r)) << 32) |
         static_cast<uint32_t>(c);
}

static inline int key_row(uint64_t key) {
  return static_cast<int>(key >> 32);
}

static inline int key_col(uint64_t key) {
  return static_cast<int>(key & 0xffffffffu);
}

template <typename F>
static std::vector<Triplet> shuffle_triplets(const std::vector<Triplet>& input,
                                            F owner,
                                            MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);

  std::vector<int> send_counts(size, 0);
  for (const auto& t : input) {
    int dest = owner(t);
    send_counts[dest]++;
  }

  std::vector<int> send_displs(size, 0);
  int total_send = 0;
  for (int i = 0; i < size; i++) {
    send_displs[i] = total_send;
    total_send += send_counts[i];
  }

  std::vector<Triplet> sendbuf(total_send);
  std::vector<int> offsets = send_displs;
  for (const auto& t : input) {
    int dest = owner(t);
    sendbuf[offsets[dest]++] = t;
  }

  std::vector<int> recv_counts(size, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, comm);

  std::vector<int> recv_displs(size, 0);
  int total_recv = 0;
  for (int i = 0; i < size; i++) {
    recv_displs[i] = total_recv;
    total_recv += recv_counts[i];
  }

  std::vector<Triplet> recvbuf(total_recv);

  std::vector<int> send_counts_bytes(size, 0), send_displs_bytes(size, 0);
  std::vector<int> recv_counts_bytes(size, 0), recv_displs_bytes(size, 0);
  const int triplet_bytes = static_cast<int>(sizeof(Triplet));
  for (int i = 0; i < size; i++) {
    send_counts_bytes[i] = send_counts[i] * triplet_bytes;
    send_displs_bytes[i] = send_displs[i] * triplet_bytes;
    recv_counts_bytes[i] = recv_counts[i] * triplet_bytes;
    recv_displs_bytes[i] = recv_displs[i] * triplet_bytes;
  }

  MPI_Alltoallv(sendbuf.empty() ? nullptr : sendbuf.data(),
                send_counts_bytes.data(),
                send_displs_bytes.data(),
                MPI_BYTE,
                recvbuf.empty() ? nullptr : recvbuf.data(),
                recv_counts_bytes.data(),
                recv_displs_bytes.data(),
                MPI_BYTE,
                comm);

  return recvbuf;
}

static void topk_inplace(std::vector<std::pair<int, double>>& v, int K) {
  if (K <= 0) {
    v.clear();
    return;
  }
  auto cmp = [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  };
  if (static_cast<int>(v.size()) > K) {
    std::nth_element(v.begin(), v.begin() + K, v.end(), cmp);
    v.resize(K);
  }
  std::sort(v.begin(), v.end(), cmp);
}

ComputeResult spgemm_topk(const std::vector<Triplet>& A_local,
                          const std::vector<Triplet>& B_local,
                          int topK,
                          MPI_Comm comm) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  (void)rank;

  double t0 = MPI_Wtime();

  // Shuffle by shared k to build partial outer products locally.
  auto A_by_k = shuffle_triplets(A_local,
                                 [size](const Triplet& t) {
                                   return owner_rank(t.c, size);
                                 },
                                 comm);
  auto B_by_k = shuffle_triplets(B_local,
                                 [size](const Triplet& t) {
                                   return owner_rank(t.r, size);
                                 },
                                 comm);

  double t_shuffle = MPI_Wtime();

  std::unordered_map<int, std::vector<std::pair<int, double>>> B_map;
  B_map.reserve(B_by_k.size());
  for (const auto& t : B_by_k) {
    B_map[t.r].push_back({t.c, t.v});
  }

  std::unordered_map<uint64_t, double> acc;
  acc.reserve(A_by_k.size());
  for (const auto& t : A_by_k) {
    auto it = B_map.find(t.c);
    if (it == B_map.end()) continue;
    const auto& bj = it->second;
    for (const auto& jb : bj) {
      acc[make_key(t.r, jb.first)] += t.v * jb.second;
    }
  }

  double t_compute = MPI_Wtime();

  std::vector<Triplet> partials;
  partials.reserve(acc.size());
  for (const auto& kv : acc) {
    if (kv.second == 0.0) continue;
    partials.push_back({key_row(kv.first), key_col(kv.first), kv.second});
  }

  // Shuffle by row i to consolidate partial sums, then compute top-K per row.
  auto row_parts = shuffle_triplets(partials,
                                    [size](const Triplet& t) {
                                      return owner_rank(t.r, size);
                                    },
                                    comm);

  std::unordered_map<uint64_t, double> row_acc;
  row_acc.reserve(row_parts.size());
  for (const auto& t : row_parts) {
    row_acc[make_key(t.r, t.c)] += t.v;
  }

  std::unordered_map<int, std::vector<std::pair<int, double>>> rows;
  rows.reserve(row_acc.size() / 4 + 1);
  for (const auto& kv : row_acc) {
    rows[key_row(kv.first)].push_back({key_col(kv.first), kv.second});
  }

  std::ostringstream oss;
  for (auto& ikv : rows) {
    auto& vec = ikv.second;
    topk_inplace(vec, topK);
    oss << ikv.first;
    for (const auto& js : vec) {
      oss << " " << js.first << ":" << js.second;
    }
    oss << "\n";
  }

  double t_row_reduce = MPI_Wtime();

  ComputeResult res;
  res.local_txt = oss.str();
  res.t_shuffle = t_shuffle - t0;
  res.t_compute = t_compute - t_shuffle;
  res.t_row_reduce = t_row_reduce - t_compute;
  return res;
}
