#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Triplet {
  int r;
  int c;
  double v;
};

static bool parse_triplet_line(const std::string& line, Triplet& t) {
  size_t p = 0;
  while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p]))) p++;
  if (p == line.size() || line[p] == '#') return false;

  std::istringstream iss(line);
  if (!(iss >> t.r >> t.c >> t.v)) return false;
  return true;
}

static std::vector<std::string> collect_paths(const std::string& path) {
  namespace fs = std::filesystem;
  std::vector<std::string> paths;
  if (fs::is_directory(path)) {
    for (auto& p : fs::directory_iterator(path)) {
      if (!p.is_regular_file()) continue;
      paths.push_back(p.path().string());
    }
    std::sort(paths.begin(), paths.end());
  } else {
    paths.push_back(path);
  }
  return paths;
}

static std::vector<Triplet> read_coo_files(const std::vector<std::string>& paths) {
  std::vector<Triplet> out;
  for (const auto& path : paths) {
    std::ifstream fin(path);
    if (!fin) {
      std::cerr << "Failed to open file: " << path << "\n";
      std::exit(1);
    }
    std::string line;
    while (std::getline(fin, line)) {
      Triplet t{};
      if (parse_triplet_line(line, t)) out.push_back(t);
    }
  }
  return out;
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

int main(int argc, char** argv) {
  std::string pathA, pathB, outPath;
  int topK = 10;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (a == "--A")
      pathA = need("--A");
    else if (a == "--B")
      pathB = need("--B");
    else if (a == "--topk")
      topK = std::stoi(need("--topk"));
    else if (a == "--out")
      outPath = need("--out");
    else if (a == "--help") {
      std::cout << "Usage: ./serial_spgemm_topk --A data/A --B data/B --topk 10 [--out out.txt]\n";
      return 0;
    }
  }

  if (pathA.empty() || pathB.empty()) {
    std::cerr << "Need --A and --B\n";
    return 3;
  }

  auto pathsA = collect_paths(pathA);
  auto pathsB = collect_paths(pathB);

  auto A_local_read = read_coo_files(pathsA);
  auto B_local_read = read_coo_files(pathsB);

  std::unordered_map<int, std::vector<std::pair<int, double>>> B_map;
  B_map.reserve(B_local_read.size());
  for (const auto& t : B_local_read) {
    B_map[t.r].push_back({t.c, t.v});
  }

  std::unordered_map<uint64_t, double> acc;
  acc.reserve(A_local_read.size());
  for (const auto& t : A_local_read) {
    auto it = B_map.find(t.c);
    if (it == B_map.end()) continue;
    const auto& bj = it->second;
    for (const auto& jb : bj) {
      acc[make_key(t.r, jb.first)] += t.v * jb.second;
    }
  }

  std::unordered_map<int, std::vector<std::pair<int, double>>> rows;
  rows.reserve(acc.size() / 4 + 1);
  for (const auto& kv : acc) {
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

  if (!outPath.empty()) {
    std::ofstream fout(outPath);
    fout << oss.str();
  } else {
    std::cout << oss.str();
  }
  return 0;
}
