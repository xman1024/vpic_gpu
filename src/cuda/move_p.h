#ifndef __CUDA_MOVE_P_H__
#define __CUDA_MOVE_P_H__

#include "../species_advance/species_advance.h"

// Assume p0 is on the device others are not
// returns and modified the same things as move_p from
// species_advance/standard/move_p.h
int cuda_move_p(particle_t* p0,
                particle_mover_t* ALIGNED(16) pm,
                accumulator_t* ALIGNED(128) a0,
                const grid_t* g,
                const float qsp);
#endif  // __CUDA_MOVE_P_H__
