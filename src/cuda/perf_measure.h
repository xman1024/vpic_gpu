#include <chrono>

#define PERF_START(x) auto start##x = std::chrono::system_clock::now();

#define PERF_END(x)                                                      \
    {                                                                    \
        auto end##x = std::chrono::system_clock::now();                  \
        std::cerr << #x << " took " << (end##x - start##x).count() / 1e9 \
                  << "s" << std::endl;                                   \
    }
