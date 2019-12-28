/*
 * Written by:
 *   Iwona Kotlarska, Łukasz Kondraciuk
 *   University of Warsaw
 *   2019 - port to CUDA for SC19 student cluster competition
 *
 */
#include <cuda_runtime.h>

#include "compute_f0_size.h"
#include "utils.h"

__global__ void compute_f0_size_kernel(const particle_t* p,
                                       int n,
                                       int* result) {
    __shared__ int res[1024];
    int tid          = threadIdx.x;
    int i            = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int loc_res      = 0;
    for (; i < n; i += stride) {
        loc_res = max(loc_res, p[i].i);
    }
    res[tid] = loc_res;
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < 1024; ++i)
            loc_res = max(loc_res, res[i]);
        *result = loc_res;
    }
}

int compute_f0_size(const particle_t* p, int n) {
    int* res;
    CUDA_CHECK(cudaMalloc((void**)&res, sizeof(int)));
    compute_f0_size_kernel<<<1, 1024>>>(p, n, res);
    int r = device_fetch_var(res) + 1;
    CUDA_CHECK(cudaFree(res));
    return r;
}
