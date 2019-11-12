#define IN_spa

#define HAS_V4_PIPELINE
#define HAS_V8_PIPELINE
#define HAS_V16_PIPELINE

#include "spa_private.h"

#include "../../../util/pipelines/pipelines_exec.h"
#include "../../../cuda/compute_f0_size.h"
#include "../../../cuda/uncenter_p.h"
#include <cuda_runtime.h>
#include "../../../cuda/utils.h"

//----------------------------------------------------------------------------//
// Reference implementation for an uncenter_p pipeline function which does not
// make use of explicit calls to vector intrinsic functions.
//----------------------------------------------------------------------------//

void uncenter_p_pipeline_scalar(center_p_pipeline_args_t* args,
                                int pipeline_rank,
                                int n_pipeline) {
    particle_t* ALIGNED(32) p;

    const float qdt_2mc   = -args->qdt_2mc;        // For backward half advance
    const float qdt_4mc   = -0.5 * args->qdt_2mc;  // For backward half rotate

    int first, n;

    // Determine which particles this pipeline processes.

    DISTRIBUTE(args->np, 16, pipeline_rank, n_pipeline, first, n);

    p = args->p0 + first;

    int f0_size = compute_f0_size(p, n);

    interpolator_t* f0;
    MALLOC(f0, f0_size);
    CUDA_CHECK(cudaMemcpy(f0, args->f0, f0_size * sizeof(interpolator_t), cudaMemcpyHostToDevice));

    uncenter_p_pipeline_cuda(p, n, f0, qdt_2mc, qdt_4mc);
}

//----------------------------------------------------------------------------//
// Top level function to select and call the proper uncenter_p pipeline
// function.
//----------------------------------------------------------------------------//

void uncenter_p_pipeline(species_t* RESTRICT sp,
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

    EXEC_PIPELINES(uncenter_p, args, 0);

    WAIT_PIPELINES();
}
