#include "advance_particle.h"
#include <iostream>
#include <vector>
#include "../util/util_base.h"
#include "debug.h"
#define IN_spa
#include "../species_advance/standard/pipeline/spa_private.h"

void run_kernel(particle_t* p0,
                int n,
                const interpolator_t* f0,
                accumulator_t* a0,
                particle_mover_t* pm,
                const grid_t* g,
                const float qdt_2mc,
                const float cdt_dx,
                const float cdt_dy,
                const float cdt_dz,
                const float qsp,
                const int max_nm,
                int* nm,
                int* skipped);

void advance_p_cuda(species_t* RESTRICT sp,
                    accumulator_array_t* RESTRICT aa,
                    const interpolator_array_t* RESTRICT ia) {
    particle_t* p0           = sp->p;
    accumulator_t* a0        = aa->a;
    const interpolator_t* f0 = ia->i;
    const grid_t* g          = ia->g;

    particle_t* p;
    particle_mover_t* pm;
    //const interpolator_t* f;

    const float qdt_2mc = (sp->q * sp->g->dt) / (2 * sp->m * sp->g->cvac);
    const float cdt_dx  = sp->g->cvac * sp->g->dt * sp->g->rdx;
    const float cdt_dy  = sp->g->cvac * sp->g->dt * sp->g->rdy;
    const float cdt_dz  = sp->g->cvac * sp->g->dt * sp->g->rdz;
    const float qsp     = sp->q;

    int nm, max_nm;

    max_nm = sp->max_nm;

    if (max_nm < 0)
        max_nm = 0;

    pm = sp->pm;
    nm = 0;

    // Process particles for this pipeline.

    int n_ignored = 0;

    run_kernel(p0, sp->np, f0, a0, sp->pm, g, qdt_2mc, cdt_dx, cdt_dy, cdt_dz, qsp,
               sp->max_nm, &nm, &n_ignored);

    sp->nm = 0;

    if (n_ignored) {
        WARNING(("Pipeline %i ran out of storage for %i movers", 0, n_ignored));
    }

    if (sp->pm + sp->nm != pm) {
        MOVE(sp->pm + sp->nm, pm, nm);
    }

    sp->nm += nm;
}
