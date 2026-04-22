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

struct ShufflePlan {
  std::vector<int> send_counts;
  std::vector<int> send_displs;
  std::vector<int> recv_counts;
  std::vector<int> recv_displs;
  std::vector<int> send_counts_bytes;
  std::vector<int> send_displs_bytes;
  std::vector<int> recv_counts_bytes;
  std::vector<int> recv_displs_bytes;
  std::vector<Triplet> sendbuf;
  std::vector<Triplet> recvbuf;
};

template <typename F>
static ShufflePlan make_shuffle_plan(const std::vector<Triplet>& input,
                                     F owner,
                                     MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);

  ShufflePlan plan;
  plan.send_counts.assign(size, 0);
  for (const auto& t : input) {
    int dest = owner(t);
    plan.send_counts[dest]++;
  }

  plan.send_displs.assign(size, 0);
  int total_send = 0;
  for (int i = 0; i < size; i++) {
    plan.send_displs[i] = total_send;
    total_send += plan.send_counts[i];
  }

  plan.sendbuf.resize(total_send);
  std::vector<int> offsets = plan.send_displs;
  for (const auto& t : input) {
    int dest = owner(t);
    plan.sendbuf[offsets[dest]++] = t;
  }

  plan.recv_counts.assign(size, 0);
  MPI_Alltoall(plan.send_counts.data(), 1, MPI_INT,
               plan.recv_counts.data(), 1, MPI_INT, comm);

  plan.recv_displs.assign(size, 0);
  int total_recv = 0;
  for (int i = 0; i < size; i++) {
    plan.recv_displs[i] = total_recv;
    total_recv += plan.recv_counts[i];
  }

  plan.recvbuf.resize(total_recv);

  plan.send_counts_bytes.assign(size, 0);
  plan.send_displs_bytes.assign(size, 0);
  plan.recv_counts_bytes.assign(size, 0);
  plan.recv_displs_bytes.assign(size, 0);
  const int triplet_bytes = static_cast<int>(sizeof(Triplet));
  for (int i = 0; i < size; i++) {
    plan.send_counts_bytes[i] = plan.send_counts[i] * triplet_bytes;
    plan.send_displs_bytes[i] = plan.send_displs[i] * triplet_bytes;
    plan.recv_counts_bytes[i] = plan.recv_counts[i] * triplet_bytes;
    plan.recv_displs_bytes[i] = plan.recv_displs[i] * triplet_bytes;
  }
  return plan;
}

template <typename F>
static std::vector<Triplet> shuffle_triplets(const std::vector<Triplet>& input,
                                             F owner,
                                             MPI_Comm comm) {
  auto plan = make_shuffle_plan(input, owner, comm);
  MPI_Alltoallv(plan.sendbuf.empty() ? nullptr : plan.sendbuf.data(),
                plan.send_counts_bytes.data(),
                plan.send_displs_bytes.data(),
                MPI_BYTE,
                plan.recvbuf.empty() ? nullptr : plan.recvbuf.data(),
                plan.recv_counts_bytes.data(),
                plan.recv_displs_bytes.data(),
                MPI_BYTE,
                comm);
  return plan.recvbuf;
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

  auto planA = make_shuffle_plan(A_local,
                                 [size](const Triplet& t) {
                                   return owner_rank(t.c, size);
                                 },
                                 comm);
  auto planB = make_shuffle_plan(B_local,
                                 [size](const Triplet& t) {
                                   return owner_rank(t.r, size);
                                 },
                                 comm);
  MPI_Request reqs[2];
  MPI_Ialltoallv(planA.sendbuf.empty() ? nullptr : planA.sendbuf.data(),
                 planA.send_counts_bytes.data(),
                 planA.send_displs_bytes.data(),
                 MPI_BYTE,
                 planA.recvbuf.empty() ? nullptr : planA.recvbuf.data(),
                 planA.recv_counts_bytes.data(),
                 planA.recv_displs_bytes.data(),
                 MPI_BYTE,
                 comm,
                 &reqs[0]);
  MPI_Ialltoallv(planB.sendbuf.empty() ? nullptr : planB.sendbuf.data(),
                 planB.send_counts_bytes.data(),
                 planB.send_displs_bytes.data(),
                 MPI_BYTE,
                 planB.recvbuf.empty() ? nullptr : planB.recvbuf.data(),
                 planB.recv_counts_bytes.data(),
                 planB.recv_displs_bytes.data(),
                 MPI_BYTE,
                 comm,
                 &reqs[1]);
  MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
  auto A_by_k = std::move(planA.recvbuf);
  auto B_by_k = std::move(planB.recvbuf);

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
