#define IN_spa

#include <cuda_runtime.h>

#include "../../../cuda/center_p.h"
#include "../../../cuda/compute_f0_size.h"
#include "../../../cuda/utils.h"
#include "../../../util/pipelines/pipelines_exec.h"
#include "spa_private.h"

//----------------------------------------------------------------------------//
// Reference implementation for a center_p pipeline function which does not
// make use of explicit calls to vector intrinsic functions.
//----------------------------------------------------------------------------//

void center_p_pipeline_scalar(center_p_pipeline_args_t* args, int n_pipeline) {
    particle_t* ALIGNED(32) p;

    const float qdt_2mc = args->qdt_2mc;
    const float qdt_4mc = 0.5 * args->qdt_2mc;  // For half Boris rotate

    int n = args->np;

    p = args->p0;

    int f0_size = compute_f0_size(p, n);

    interpolator_t* f0;
    CUDA_CHECK(cudaMalloc((void**)&f0, sizeof(interpolator_t) * f0_size));
    CUDA_CHECK(cudaMemcpy(f0, args->f0, f0_size * sizeof(interpolator_t),
                          cudaMemcpyHostToDevice));

    center_p_pipeline_cuda(p, n, f0, qdt_2mc, qdt_4mc);
    CUDA_CHECK(cudaFree(f0));
}

//----------------------------------------------------------------------------//
// Top level function to select and call the proper center_p pipeline
// function.
//----------------------------------------------------------------------------//

void center_p_pipeline(species_t* RESTRICT sp,
                       const interpolator_array_t* RESTRICT ia) {
    DECLARE_ALIGNED_ARRAY(center_p_pipeline_args_t, 128, args, 1);

    if (!sp || !ia || sp->g != ia->g) {
        ERROR(("Bad args"));
    }

    // Have the pipelines do the bulk of particles in blocks and have the
    // host do the final incomplete block.

    args->p0      = sp->device_p0;
    args->f0      = ia->i;
    args->qdt_2mc = (sp->q * sp->g->dt) / (2 * sp->m * sp->g->cvac);
    args->np      = sp->np;

    center_p_pipeline_scalar(args, 0);
}
