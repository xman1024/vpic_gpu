#include <cuda_runtime.h>

#include "energy_p.h"
#include "utils.h"

__global__ void energy_p_kernel(const particle_t* p,
                                int n0,
                                interpolator_t* f,
                                const float qdt_2mc,
                                const float msp,
                                double* en_buffer) {
    const float one = 1.0;

    float dx, dy, dz;
    float v0, v1, v2;
    __shared__ double res[1024];
    double en        = 0;
    int tid          = threadIdx.x;
    int n            = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    int i;

    for (; n < n0; n += stride) {
        dx = p[n].dx;
        dy = p[n].dy;
        dz = p[n].dz;
        i  = p[n].i;

        v0 = p[n].ux + qdt_2mc * ((f[i].ex + dy * f[i].dexdy) +
                                  dz * (f[i].dexdz + dy * f[i].d2exdydz));

        v1 = p[n].uy + qdt_2mc * ((f[i].ey + dz * f[i].deydz) +
                                  dx * (f[i].deydx + dz * f[i].d2eydzdx));

        v2 = p[n].uz + qdt_2mc * ((f[i].ez + dx * f[i].dezdx) +
                                  dy * (f[i].dezdy + dx * f[i].d2ezdxdy));

        v0 = v0 * v0 + v1 * v1 + v2 * v2;

        v0 = (msp * p[n].w) * (v0 / (one + sqrtf(one + v0)));

        en += (double)v0;
    }
    res[tid] = en;
    __syncthreads();
    if (tid == 0) {
        en = 0;
        for (int i = 0; i < 1024; ++i)
            en += res[i];
        en_buffer[blockIdx.x] = en;
    }
}

double energy_p_pipeline_cuda(const particle_t* p,
                              int n,
                              interpolator_t* f0,
                              const float qdt_2mc,
                              const float msp) {
    double* en_buffer = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&en_buffer, sizeof(double) * 1024));
    energy_p_kernel<<<1024, 1024>>>(p, n, f0, qdt_2mc, msp, en_buffer);
    double host_buffer[1024];
    CUDA_CHECK(cudaMemcpy(host_buffer, en_buffer, 1024 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(en_buffer));

    double res = 0;
    for (int i = 0; i < 1024; ++i)
        res += host_buffer[i];

    return res;
}
