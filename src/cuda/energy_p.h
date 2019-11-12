#ifndef __CUDA_ENERGY_P_H__
#define __CUDA_ENERGY_P_H__

#include "../species_advance/species_advance.h"

void energy_p_pipeline_cuda(const particle_t* p, int n, interpolator_t* f0, const float qdt_2mc, const float msp, double* en);

#endif //  __CUDA_ENERGY_P_H__

