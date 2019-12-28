/*
 * Written by:
 *   Iwona Kotlarska, Łukasz Kondraciuk
 *   University of Warsaw
 *   2019 - port to CUDA for SC19 student cluster competition
 *
 */
#ifndef __CUDA_COMPUTE_F0_H__
#define __CUDA_COMPUTE_F0_H__

#include "../species_advance/species_advance.h"

int compute_f0_size(const particle_t* p, int n);

#endif  // __CUDA_COMPUTE_F0_H__
