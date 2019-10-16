#ifndef __ADVANCE_PARTICLE__H__
#define __ADVANCE_PARTICLE__H__
#include "../species_advance/species_advance.h"

void advance_p_cuda(species_t* RESTRICT sp,
                    accumulator_array_t* RESTRICT aa,
                    const interpolator_array_t* RESTRICT ia);

void cuda_move_p(particle_t* p0,
                 particle_mover_t* pm,
                 int n,
                 accumulator_t* a0,
                 int64_t* neighbours,
                 const float qsp,
                 int* nm,
                 particle_mover_t* pm_save,
                 int64_t rangeh,
                 int64_t rangel);

#endif  // __ADVANCE_PARTICLE__H__