#include "uncenter_p.h"
#include "utils.h"
#include <cuda_runtime.h>

__global__ void uncenter_p_kernel(particle_t* p, int n, interpolator_t* f0, const float qdt_2mc, const float qdt_4mc)
{
    const float one       = 1.0;
    const float one_third = 1.0 / 3.0;
    const float two_fifteenths = 2.0 / 15.0;

    float dx, dy, dz, ux, uy, uz;
    float hax, hay, haz, cbx, cby, cbz;
    float v0, v1, v2, v3, v4;
    int ii;

    interpolator_t* f;

    // Process particles for this pipeline.

    for (; n; n--, p++) {
        dx = p->dx;  // Load position
        dy = p->dy;
        dz = p->dz;
        ii = p->i;

        f = f0 + ii;  // Interpolate E

        hax = qdt_2mc *
              ((f->ex + dy * f->dexdy) + dz * (f->dexdz + dy * f->d2exdydz));

        hay = qdt_2mc *
              ((f->ey + dz * f->deydz) + dx * (f->deydx + dz * f->d2eydzdx));

        haz = qdt_2mc *
              ((f->ez + dx * f->dezdx) + dy * (f->dezdy + dx * f->d2ezdxdy));

        cbx = f->cbx + dx * f->dcbxdx;  // Interpolate B
        cby = f->cby + dy * f->dcbydy;
        cbz = f->cbz + dz * f->dcbzdz;

        ux = p->ux;  // Load momentum
        uy = p->uy;
        uz = p->uz;

        v0 = qdt_4mc / (float)sqrt(one + (ux * ux + (uy * uy + uz * uz)));
        /**/  // Boris - scalars
        v1 = cbx * cbx + (cby * cby + cbz * cbz);
        v2 = (v0 * v0) * v1;
        v3 = v0 * (one + v2 * (one_third + v2 * two_fifteenths));
        v4 = v3 / (one + v1 * (v3 * v3));
        v4 += v4;

        v0 = ux + v3 * (uy * cbz - uz * cby);  // Boris - uprime
        v1 = uy + v3 * (uz * cbx - ux * cbz);
        v2 = uz + v3 * (ux * cby - uy * cbx);

        ux += v4 * (v1 * cbz - v2 * cby);  // Boris - rotation
        uy += v4 * (v2 * cbx - v0 * cbz);
        uz += v4 * (v0 * cby - v1 * cbx);

        ux += hax;  // Half advance E
        uy += hay;
        uz += haz;

        p->ux = ux;  // Store momentum
        p->uy = uy;
        p->uz = uz;
    }
}

void center_p_pipeline_cuda(particle_t* p, int n, interpolator_t* f0, const float qdt_2mc, const float qdt_4mc)
{
    uncenter_p_kernel<<<1, 1>>>(p, n, f0, qdt_2mc, qdt_4mc);
}
