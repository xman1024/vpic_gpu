#include <cuda_runtime.h>

#include "rho_p.h"
#include "utils.h"

__global__ void rho_p_kernel(const particle_t* p,
                             const float q_8V,
                             const int np,
                             const int sy,
                             const int sz,
                             field_t* f) {
    float w0, w1, w2, w3, w4, w5, w6, w7, dz;

    int v;

    int n            = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (; n < np; n += stride) {
        // After detailed experiments and studying of assembly dumps, it was
        // determined that if the platform does not support efficient 4-vector
        // SIMD memory gather/scatter operations, the savings from using
        // "trilinear" are slightly outweighed by the overhead of the
        // gather/scatters.

        // Load the particle data

        w0 = p[n].dx;
        w1 = p[n].dy;
        dz = p[n].dz;
        v  = p[n].i;
        w7 = p[n].w * q_8V;

        // Compute the trilinear weights
        // Though the PPE should have hardware fma/fmaf support, it was
        // measured to be more efficient _not_ to use it here.  (Maybe the
        // compiler isn't actually generating the assembly for it.

#define FMA(x, y, z) ((z) + (x) * (y))
#define FNMS(x, y, z) ((z) - (x) * (y))
        w6 = FNMS(w0, w7, w7);  // q(1-dx)
        w7 = FMA(w0, w7, w7);   // q(1+dx)
        w4 = FNMS(w1, w6, w6);
        w5 = FNMS(w1, w7, w7);  // q(1-dx)(1-dy), q(1+dx)(1-dy)
        w6 = FMA(w1, w6, w6);
        w7 = FMA(w1, w7, w7);  // q(1-dx)(1+dy), q(1+dx)(1+dy)
        w0 = FNMS(dz, w4, w4);
        w1 = FNMS(dz, w5, w5);
        w2 = FNMS(dz, w6, w6);
        w3 = FNMS(dz, w7, w7);
        w4 = FMA(dz, w4, w4);
        w5 = FMA(dz, w5, w5);
        w6 = FMA(dz, w6, w6);
        w7 = FMA(dz, w7, w7);
#undef FNMS
#undef FMA

        // Reduce the particle charge to rhof

        atomicAdd(&f[v].rhof, w0);
        atomicAdd(&f[v + 1].rhof, w1);
        atomicAdd(&f[v + sy].rhof, w2);
        atomicAdd(&f[v + sy + 1].rhof, w3);
        atomicAdd(&f[v + sz].rhof, w4);
        atomicAdd(&f[v + sz + 1].rhof, w5);
        atomicAdd(&f[v + sz + sy].rhof, w6);
        atomicAdd(&f[v + sz + sy + 1].rhof, w7);
    }
}

void rho_p_cuda(const particle_t* p,
                const float q_8V,
                const int np,
                const int sy,
                const int sz,
                field_t* f) {
    rho_p_kernel<<<1024, 1024>>>(p, q_8V, np, sy, sz, f);
}
