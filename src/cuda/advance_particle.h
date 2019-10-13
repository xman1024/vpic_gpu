#ifndef __ADVANCE_PARTICLE__H__
#define __ADVANCE_PARTICLE__H__
#include "../species_advance/species_advance.h"

void advance_p_cuda(species_t* RESTRICT sp,
                    accumulator_array_t* RESTRICT aa,
                    const interpolator_array_t* RESTRICT ia);

#endif  // __ADVANCE_PARTICLE__H__