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
    particle_t* ALIGNED(128) p0           = sp->p;
    accumulator_t* ALIGNED(128) a0        = aa->a;
    const interpolator_t* ALIGNED(128) f0 = ia->i;
    const grid_t* g                       = ia->g;

    particle_t* ALIGNED(32) p;
    particle_mover_t* ALIGNED(16) pm;
    const interpolator_t* ALIGNED(16) f;
    float* ALIGNED(16) a;

    const float qdt_2mc = (sp->q * sp->g->dt) / (2 * sp->m * sp->g->cvac);
    const float cdt_dx  = sp->g->cvac * sp->g->dt * sp->g->rdx;
    const float cdt_dy  = sp->g->cvac * sp->g->dt * sp->g->rdy;
    const float cdt_dz  = sp->g->cvac * sp->g->dt * sp->g->rdz;
    const float qsp     = sp->q;

    int itmp, n, nm, max_nm;

    DECLARE_ALIGNED_ARRAY(particle_mover_t, 1, local_pm, 1);

    n = sp->np;
    // Determine which movers are reserved for this pipeline.
    // Movers (16 bytes) should be reserved for pipelines in at least
    // multiples of 8 such that the set of particle movers reserved for
    // a pipeline is 128-byte aligned and a multiple of 128-byte in
    // size.  The host is guaranteed to get enough movers to process its
    // particles with this allocation.

    max_nm = sp->max_nm - (sp->np & 15);

    if (max_nm < 0)
        max_nm = 0;

    itmp   = 0;
    max_nm = sp->max_nm - itmp;

    pm   = sp->pm + itmp;
    nm   = 0;
    itmp = 0;

    // Determine which accumulator array to use
    // The host gets the first accumulator array.

    //   if ( pipeline_rank != n_pipeline )
    //     a0 += ( 1 + pipeline_rank ) *
    //           POW2_CEIL( (args->nx+2)*(args->ny+2)*(args->nz+2), 2 );

    // Process particles for this pipeline.

    run_kernel(p0, n, f0, a0, sp->pm, g, qdt_2mc, cdt_dx, cdt_dy, cdt_dz, qsp, max_nm,
               &nm, &itmp);

    int n_ignored = itmp;
    sp->nm        = 0;

    if (n_ignored) {
        WARNING(("Pipeline %i ran out of storage for %i movers", 0, n_ignored));
    }

    if (sp->pm + sp->nm != pm) {
        MOVE(sp->pm + sp->nm, pm, nm);
    }

    sp->nm += nm;
}
