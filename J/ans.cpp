#include <iostream>
#include <vector>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <array>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sched.h>
#include <thread>
#include <atomic>
#include <omp.h>

/* Power monitor helpers */
int g_sock = -1;

void init_power_client(const char* ip, int port) {
    if ((g_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        exit(1);
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        exit(1);
    }

    if (connect(g_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        exit(1);
    }
}

std::pair<double, double> get_remote_power() {
    if (g_sock < 0) return {0.0, 0.0};

    const char* msg = "GET\n";
    send(g_sock, msg, strlen(msg), 0);

    char buffer[128] = {0};
    int valread = read(g_sock, buffer, 128);

    if (valread > 0) {
        double cpu, other;
        sscanf(buffer, "%lf %lf", &cpu, &other);
        return {cpu, other};
    }
    return {0.0, 0.0};
}

struct MappedFile {
    int fd;
    void* data;
    size_t size;
    MappedFile(const char* fn) {
        fd = open(fn, O_RDONLY);
        if (fd < 0) { perror("open"); exit(1); }
        struct stat sb;
        fstat(fd, &sb);
        size = sb.st_size;
        data = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (data == MAP_FAILED) { perror("mmap"); exit(1); }
    }
    ~MappedFile() { munmap(data, size); close(fd); }
};

std::vector<int> parse_cpu_range(const std::string& s) {
    std::vector<int> cpus;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        std::string part = s.substr(start, comma - start);

        size_t dash = part.find('-');
        if (dash != std::string::npos) {
            int a = std::stoi(part.substr(0, dash));
            int b = std::stoi(part.substr(dash + 1));
            for (int i = a; i <= b; ++i) cpus.push_back(i);
        } else {
            cpus.push_back(std::stoi(part));
        }
        start = comma + 1;
    }
    return cpus;
}

void fix_cpu_affinity() {
    FILE* f = fopen("/sys/fs/cgroup/cpuset.cpus", "r");
    if (!f) f = fopen("/sys/fs/cgroup/cpuset.cpus.effective", "r");
    if (!f) f = fopen("/sys/fs/cgroup/cpuset/cpuset.cpus", "r");

    if (!f) {
        fprintf(stderr, "[AFFINITY] Cannot read cgroup cpuset\n");
        return;
    }

    char buf[256];
    if (!fgets(buf, sizeof(buf), f)) {
        fclose(f);
        fprintf(stderr, "[AFFINITY] Cannot read cpuset content\n");
        return;
    }
    fclose(f);
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

    fprintf(stderr, "[AFFINITY] Cgroup cpuset: %s\n", buf);

    std::vector<int> cpus = parse_cpu_range(buf);
    fprintf(stderr, "[AFFINITY] Parsed %zu CPUs\n", cpus.size());

    if (cpus.empty()) return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cpus) {
        CPU_SET(cpu, &cpuset);
    }

    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        fprintf(stderr, "[AFFINITY] Successfully set affinity to %zu CPUs\n", cpus.size());
    } else {
        perror("[AFFINITY] sched_setaffinity failed");
    }
}

const int PATTERN_LEN = 64;
const uint64_t BASE1 = 11400714819323198485ULL;
const uint64_t BASE2 = 14029467366897019727ULL;
const double POWER_LIMIT = 600.0;

struct Entry {
    uint64_t h1;
    uint64_t h2;
    uint32_t count;
};

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t hash_key(uint64_t h1, uint64_t h2) {
    uint64_t x = h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    return mix64(x);
}

class HashTable {
public:
    explicit HashTable(size_t expected) {
        size_t cap = 1;
        while (cap < expected * 2) cap <<= 1;
        table.assign(cap, Entry{0, 0, 0});
        mask = cap - 1;
    }

    void add(uint64_t h1, uint64_t h2) {
        size_t idx = hash_key(h1, h2) & mask;
        while (true) {
            Entry& e = table[idx];
            if (e.count == 0) {
                e.h1 = h1;
                e.h2 = h2;
                e.count = 1;
                return;
            }
            if (e.h1 == h1 && e.h2 == h2) {
                e.count++;
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    uint32_t get(uint64_t h1, uint64_t h2) const {
        size_t idx = hash_key(h1, h2) & mask;
        while (true) {
            const Entry& e = table[idx];
            if (e.count == 0) return 0;
            if (e.h1 == h1 && e.h2 == h2) return e.count;
            idx = (idx + 1) & mask;
        }
    }

private:
    std::vector<Entry> table;
    size_t mask;
};

static inline uint64_t char_val(uint8_t c) {
    return static_cast<uint64_t>(c - 'a' + 1);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <text_file> <pattern_file> <output_file>" << std::endl;
        return 1;
    }

    fix_cpu_affinity();
    init_power_client("127.0.0.1", 19937);

    MappedFile f_pat(argv[2]);
    const uint8_t* p_ptr = (const uint8_t*)f_pat.data;
    uint32_t K = *(uint32_t*)p_ptr;
    p_ptr += 4;

    std::vector<const uint8_t*> patterns;
    patterns.reserve(K);
    for (uint32_t k = 0; k < K; ++k) {
        patterns.push_back(p_ptr + k * PATTERN_LEN);
    }

    MappedFile f_text(argv[1]);
    const uint8_t* t_ptr = (const uint8_t*)f_text.data;
    uint64_t N = *(uint64_t*)t_ptr;
    const uint8_t* text = t_ptr + 8;

    std::array<uint64_t, PATTERN_LEN> pow1;
    std::array<uint64_t, PATTERN_LEN> pow2;
    pow1[PATTERN_LEN - 1] = 1;
    pow2[PATTERN_LEN - 1] = 1;
    for (int i = PATTERN_LEN - 2; i >= 0; --i) {
        pow1[i] = pow1[i + 1] * BASE1;
        pow2[i] = pow2[i + 1] * BASE2;
    }

    uint64_t powL1 = 1;
    uint64_t powL2 = 1;
    for (int i = 0; i < PATTERN_LEN; ++i) {
        powL1 *= BASE1;
        powL2 *= BASE2;
    }

    size_t expected = static_cast<size_t>(K) * (1 + PATTERN_LEN * 25);
    HashTable table(expected);

    for (uint32_t k = 0; k < K; ++k) {
        const uint8_t* pat = patterns[k];
        uint64_t h1 = 0, h2 = 0;
        for (int i = 0; i < PATTERN_LEN; ++i) {
            uint64_t v = char_val(pat[i]);
            h1 = h1 * BASE1 + v;
            h2 = h2 * BASE2 + v;
        }
        table.add(h1, h2);

        for (int i = 0; i < PATTERN_LEN; ++i) {
            uint64_t oldv = char_val(pat[i]);
            for (uint64_t newv = 1; newv <= 26; ++newv) {
                if (newv == oldv) continue;
                int64_t delta = static_cast<int64_t>(newv) - static_cast<int64_t>(oldv);
                uint64_t h1v = h1 + static_cast<uint64_t>(delta) * pow1[i];
                uint64_t h2v = h2 + static_cast<uint64_t>(delta) * pow2[i];
                table.add(h1v, h2v);
            }
        }
    }

    std::atomic<bool> stop_monitor{false};
    std::atomic<bool> throttle{false};
    std::thread monitor([&]() {
        using namespace std::chrono;
        while (!stop_monitor.load(std::memory_order_relaxed)) {
            auto p = get_remote_power();
            double total = p.first + p.second;
            if (total > POWER_LIMIT) {
                throttle.store(true, std::memory_order_relaxed);
            } else {
                throttle.store(false, std::memory_order_relaxed);
            }
            std::this_thread::sleep_for(milliseconds(20));
        }
    });

    uint64_t total_matches = 0;
    if (N >= PATTERN_LEN) {
        uint64_t num_windows = N - PATTERN_LEN + 1;
        const uint64_t CHECK_INTERVAL = 1u << 14;

        #pragma omp parallel reduction(+:total_matches)
        {
            int tid = omp_get_thread_num();
            int threads = omp_get_num_threads();
            uint64_t start = (num_windows * tid) / threads;
            uint64_t end = (num_windows * (tid + 1)) / threads;

            if (start < end) {
                uint64_t h1 = 0, h2 = 0;
                const uint8_t* w = text + start;
                for (int j = 0; j < PATTERN_LEN; ++j) {
                    uint64_t v = char_val(w[j]);
                    h1 = h1 * BASE1 + v;
                    h2 = h2 * BASE2 + v;
                }

                uint64_t i = start;
                for (; i + 1 < end; ++i) {
                    total_matches += table.get(h1, h2);
                    if (((i - start) & (CHECK_INTERVAL - 1)) == 0) {
                        if (throttle.load(std::memory_order_relaxed)) {
                            std::this_thread::sleep_for(std::chrono::microseconds(200));
                        }
                    }
                    uint64_t oldv = char_val(text[i]);
                    uint64_t newv = char_val(text[i + PATTERN_LEN]);
                    h1 = h1 * BASE1 + newv - oldv * powL1;
                    h2 = h2 * BASE2 + newv - oldv * powL2;
                }
                total_matches += table.get(h1, h2);
            }
        }
    }

    stop_monitor.store(true, std::memory_order_relaxed);
    monitor.join();

    int out_fd = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd >= 0) {
        write(out_fd, &total_matches, sizeof(total_matches));
        close(out_fd);
    } else {
        perror("Output file open failed");
    }

    if (g_sock >= 0) close(g_sock);
    return 0;
}
