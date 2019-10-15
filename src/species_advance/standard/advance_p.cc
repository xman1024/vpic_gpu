#define IN_spa

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include "../../cuda/advance_particle.h"
#include "../../cuda/debug.h"
#include "../../cuda/perf_measure.h"
#include "../../cuda/utils.h"
#include "../species_advance.h"
//----------------------------------------------------------------------------//
// Top level function to select and call particle advance function using the
// desired particle advance abstraction.  Currently, the only abstraction
// available is the pipeline abstraction.
//----------------------------------------------------------------------------//

double abs_dif(particle_t* p1, particle_t* p2, size_t count) {
    double res = 0.;
    for (size_t i = 0; i < count; ++i) {
        res += diff(p1[i], p2[i]);
    }
    return res;
}

void advance_p(species_t* RESTRICT sp,
               accumulator_array_t* RESTRICT aa,
               const interpolator_array_t* RESTRICT ia) {
    PERF_START(advance_p)
    species_t sp2 = *sp;
    DEBUG(aa->stride)

    sp2.p = new particle_t[sp2.np];
    // CUDA_CHECK(cudaMallocHost((void**)&sp2.p, sp2.np * sizeof(particle_t)));

    memcpy(sp2.p, sp->p, sp->np * sizeof(particle_t));

    PERF_START(cuda_advance_p)
    advance_p_cuda(&sp2, aa, ia);
    PERF_END(cuda_advance_p)

    PERF_START(cpu_advance_p)
    advance_p_pipeline(sp, aa, ia);
    PERF_END(cpu_advance_p)

    double tot_difference = abs_dif(sp2.p, sp->p, sp->np);
    std::cerr << std::setprecision(13) << "!!!! Total difference = " << tot_difference
              << " avg difference = " << tot_difference / sp->np << std::endl;

    delete[] sp2.p;
    // CUDA_CHECK(cudaFreeHost(sp2.p));
    PERF_END(advance_p)
}
