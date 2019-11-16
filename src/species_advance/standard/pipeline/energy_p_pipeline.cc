#define IN_spa

#define HAS_V4_PIPELINE
#define HAS_V8_PIPELINE
#define HAS_V16_PIPELINE

#include <cuda_runtime.h>

#include "../../../cuda/compute_f0_size.h"
#include "../../../cuda/energy_p.h"
#include "../../../cuda/utils.h"
#include "../../../util/pipelines/pipelines_exec.h"
#include "spa_private.h"

//----------------------------------------------------------------------------//
// Reference implementation for an energy_p pipeline function which does not
// make use of explicit calls to vector intrinsic functions.  This function
// calculates kinetic energy, normalized by c^2.
//----------------------------------------------------------------------------//

void energy_p_pipeline_scalar(energy_p_pipeline_args_t* RESTRICT args,
                              int pipeline_rank,
                              int n_pipeline) {
    const particle_t* p = args->p;

    const float qdt_2mc = args->qdt_2mc;
    const float msp     = args->msp;

    int n0, n1;

    // Determine which particles this pipeline processes.

    DISTRIBUTE(args->np, 16, pipeline_rank, n_pipeline, n0, n1);

    interpolator_t* f;
    int f0_size = compute_f0_size(p + n0, n1);
    CUDA_CHECK(cudaMalloc((void**)&f, f0_size * sizeof(interpolator_t)));
    CUDA_CHECK(cudaMemcpy(f, args->f, f0_size * sizeof(interpolator_t),
                          cudaMemcpyHostToDevice));

    // Process particles quads for this pipeline.

    args->en[pipeline_rank] =
        energy_p_pipeline_cuda(p + n0, n1, f, qdt_2mc, msp);

    CUDA_CHECK(cudaFree(f));
}

//----------------------------------------------------------------------------//
// Top level function to select and call the proper energy_p pipeline
// function.
//----------------------------------------------------------------------------//

double energy_p_pipeline(const species_t* RESTRICT sp,
                         const interpolator_array_t* RESTRICT ia) {
    DECLARE_ALIGNED_ARRAY(energy_p_pipeline_args_t, 128, args, 1);

    DECLARE_ALIGNED_ARRAY(double, 128, en, MAX_PIPELINE + 1);

    double local, global;
    int rank;

    if (!sp || !ia || sp->g != ia->g) {
        ERROR(("Bad args"));
    }

    // Have the pipelines do the bulk of particles in blocks and have the
    // host do the final incomplete block.

    args->p       = sp->device_p0;
    args->f       = ia->i;
    args->en      = en;
    args->qdt_2mc = (sp->q * sp->g->dt) / (2 * sp->m * sp->g->cvac);
    args->msp     = sp->m;
    args->np      = sp->np;

    EXEC_PIPELINES(energy_p, args, 0);

    WAIT_PIPELINES();

    local = 0.0;
    for (rank = 0; rank <= N_PIPELINE; rank++) {
        local += en[rank];
    }

    mp_allsum_d(&local, &global, 1);

    return global * ((double)sp->g->cvac * (double)sp->g->cvac);
}
