#ifndef _vacuum_compute_rhob_pipeline_h_
#define _vacuum_compute_rhob_pipeline_h_

#ifndef IN_vacuum_compute_rhob_pipeline
#error \
    "Only include vacuum_compute_rhob_pipeline.h in vacuum_compute_rhob_pipeline source files."
#endif

#include "../sfa_private.h"

typedef struct pipeline_args {
    /**/ field_t* ALIGNED(128) f;
    const sfa_params_t* p;
    const grid_t* g;
} pipeline_args_t;

#define DECLARE_STENCIL()                                       \
    /**/ field_t* ALIGNED(128) f                   = args->f;     \
    const material_coefficient_t* ALIGNED(128) m = args->p->mc; \
    const grid_t* g                              = args->g;     \
    const int nx = g->nx, ny = g->ny, nz = g->nz;               \
                                                                \
    const float nc = m->nonconductive;                          \
    const float px = (nx > 1) ? g->eps0 * m->epsx * g->rdx : 0; \
    const float py = (ny > 1) ? g->eps0 * m->epsy * g->rdy : 0; \
    const float pz = (nz > 1) ? g->eps0 * m->epsz * g->rdz : 0; \
                                                                \
    field_t* ALIGNED(16) f0;                                    \
    field_t *ALIGNED(16) fx, *ALIGNED(16) fy, *ALIGNED(16) fz;  \
    int x, y, z

#define f(x, y, z) f[VOXEL(x, y, z, nx, ny, nz)]

#define INIT_STENCIL()    \
    f0 = &f(x, y, z);     \
    fx = &f(x - 1, y, z); \
    fy = &f(x, y - 1, z); \
    fz = &f(x, y, z - 1)

#define NEXT_STENCIL()  \
    f0++;               \
    fx++;               \
    fy++;               \
    fz++;               \
    x++;                \
    if (x > nx) {       \
        /**/ y++;         \
        x = 2;          \
        if (y > ny)     \
            z++;        \
        if (y > ny)     \
            y = 2;      \
        INIT_STENCIL(); \
    }

#define UPDATE_DERR_E()                                                \
    f0->rhob = nc * (px * (f0->ex - fx->ex) + py * (f0->ey - fy->ey) + \
                     pz * (f0->ez - fz->ez) - f0->rhof)

void vacuum_compute_rhob_pipeline_scalar(pipeline_args_t* args,
                                         int pipeline_rank,
                                         int n_pipeline);

#endif  // _vacuum_compute_rhob_pipeline_h_
