#include <cuda_runtime.h>

#include "compute_f0_size.h"
#include "utils.h"

__global__ void compute_f0_size_kernel(const particle_t* p,
                                       int n,
                                       int* result) {
    int res = 0;
    for (; n; n--, p++) {
        if (p->i > res)
            res = p->i;
    }
    *result = res;
}

int compute_f0_size(const particle_t* p, int n) {
    int* res;
    CUDA_CHECK(cudaMalloc((void**)&res, sizeof(int)));
    compute_f0_size_kernel<<<1, 1>>>(p, n, res);
    int r = device_fetch_var(res) + 1;
    CUDA_CHECK(cudaFree(res));
    return r;
}
