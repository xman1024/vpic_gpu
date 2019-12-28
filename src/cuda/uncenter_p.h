/*
 * Written by:
 *   Iwona Kotlarska, Łukasz Kondraciuk
 *   University of Warsaw
 *   2019 - port to CUDA for SC19 student cluster competition
 *
 */
#ifndef __CUDA_UNCENTER_P_H__
#define __CUDA_UNCENTER_P_H__

#include "../species_advance/species_advance.h"

void uncenter_p_pipeline_cuda(particle_t* p,
                              int n,
                              interpolator_t* f0,
                              const float qdt_2mc,
                              const float qdt_4mc);

#endif  //  __CUDA_UNCENTER_P_H__
