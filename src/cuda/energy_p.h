/*
 * Written by:
 *   Iwona Kotlarska, Łukasz Kondraciuk
 *   University of Warsaw
 *   2019 - port to CUDA for SC19 student cluster competition
 *
 */
#ifndef __CUDA_ENERGY_P_H__
#define __CUDA_ENERGY_P_H__

#include "../species_advance/species_advance.h"

double energy_p_pipeline_cuda(const particle_t* p,
                              int n,
                              interpolator_t* f0,
                              const float qdt_2mc,
                              const float msp);

#endif  //  __CUDA_ENERGY_P_H__
