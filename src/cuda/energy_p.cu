#include "energy_p.h"
#include "utils.h"
#include <cuda_runtime.h>

__global__ void energy_p_kernel(const particle_t* p, int n0, interpolator_t* f, const float qdt_2mc, const float msp, double* en)
{
    const float one     = 1.0;

    float dx, dy, dz;
    float v0, v1, v2;
    int i;

    for (int n = 0; n < n0; n++) {
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

        *en += (double)v0;
    }
}

void energy_p_pipeline_cuda(const particle_t* p, int n, interpolator_t* f0, const float qdt_2mc, const float msp, double* en)
{
    energy_p_kernel<<<1, 1>>>(p, n, f0, qdt_2mc, msp, en);
}
