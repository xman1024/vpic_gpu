/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Revised and extended from earlier V4PIC versions
 *
 */

#define IN_spa

#include "../species_advance.h"
#include "../../cuda/compute_f0_size.h"
#include "../../cuda/rho_p.h"
#include <cuda_runtime.h>
#include "../../cuda/utils.h"

// accumulate_rho_p adds the charge density associated with the
// supplied particle array to the rhof of the fields.  Trilinear
// interpolation is used.  rhof is known at the nodes at the same time
// as particle positions.  No effort is made to fix up edges of the
// computational domain; see note in synchronize_rhob about why this
// is done this way.  All particles on the list must be inbounds.

void accumulate_rho_p(/**/ field_array_t* RESTRICT fa,
                      const species_t* RESTRICT sp) {
    if (!fa || !sp || fa->g != sp->g)
        ERROR(("Bad args"));

    const particle_t* p = sp->device_p0;

    const float q_8V = sp->q * sp->g->r8V;
    const int np     = sp->np;
    const int sy     = sp->g->sy;
    const int sz     = sp->g->sz;

    int f_size = compute_f0_size(p, np);
#define MAXIMIZE(a, b) a = a > b ? a : b
    MAXIMIZE(f_size, f_size);
    MAXIMIZE(f_size, f_size + 1);
    MAXIMIZE(f_size, f_size + sy);
    MAXIMIZE(f_size, f_size + sy + 1);
    MAXIMIZE(f_size, f_size + sz);
    MAXIMIZE(f_size, f_size + sz + 1);
    MAXIMIZE(f_size, f_size + sz + sy);
    MAXIMIZE(f_size, f_size + sz + sy + 1);
#undef MAXIMIZE

    field_t* f;
    CUDA_CHECK(cudaMalloc((void**)&f, sizeof(field_t) * f_size));
    CUDA_CHECK(cudaMemcpy(f, fa->f, sizeof(field_t) * f_size, cudaMemcpyHostToDevice));

    rho_p_cuda(p, q_8V, np, sy, sz, f);

    CUDA_CHECK(cudaMemcpy(fa->f, f, sizeof(field_t) * f_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(f));
}

// assume p is on CPU
void accumulate_rhob(field_t* RESTRICT ALIGNED(128) f,
                     const particle_t* RESTRICT ALIGNED(32) p,
                     const grid_t* RESTRICT g,
                     const float qsp) {

    // See note in rhof for why this variant is used.
    float w0 = p->dx, w1 = p->dy, w2, w3, w4, w5, w6, w7, dz = p->dz;
    int v = p->i, x, y, z, sy = g->sy, sz = g->sz;
    w7 = (qsp * g->r8V) * p->w;

    // Compute the trilinear weights
    // See note in rhof for why FMA and FNMS are done this way.

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

    // Adjust the weights for a corrected local accumulation of rhob.
    // See note in synchronize_rho why we must do this for rhob and not
    // for rhof.

    x = v;
    z = x / sz;
    if (z == 1)
        w0 += w0, w1 += w1, w2 += w2, w3 += w3;
    if (z == g->nz)
        w4 += w4, w5 += w5, w6 += w6, w7 += w7;
    x -= sz * z;
    y = x / sy;
    if (y == 1)
        w0 += w0, w1 += w1, w4 += w4, w5 += w5;
    if (y == g->ny)
        w2 += w2, w3 += w3, w6 += w6, w7 += w7;
    x -= sy * y;
    if (x == 1)
        w0 += w0, w2 += w2, w4 += w4, w6 += w6;
    if (x == g->nx)
        w1 += w1, w3 += w3, w5 += w5, w7 += w7;

    // Reduce the particle charge to rhob

    f[v].rhob += w0;
    f[v + 1].rhob += w1;
    f[v + sy].rhob += w2;
    f[v + sy + 1].rhob += w3;
    f[v + sz].rhob += w4;
    f[v + sz + 1].rhob += w5;
    f[v + sz + sy].rhob += w6;
    f[v + sz + sy + 1].rhob += w7;
}
