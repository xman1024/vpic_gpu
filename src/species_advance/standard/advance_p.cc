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

void advance_p(species_t* RESTRICT sp,
               accumulator_array_t* RESTRICT aa,
               const interpolator_array_t* RESTRICT ia) {
    advance_p_cuda(sp, aa, ia);
}
