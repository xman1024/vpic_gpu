/*
 * Written by:
 *   Iwona Kotlarska, Łukasz Kondraciuk
 *   University of Warsaw
 *   2019 - port to CUDA for SC19 student cluster competition
 *
 */
#ifndef __CUDA_RHO_P_H__
#define __CUDA_RHO_P_H__

#include "../species_advance/species_advance.h"

void rho_p_cuda(const particle_t* p,
                const float q_8V,
                const int np,
                const int sy,
                const int sz,
                field_t* f);

#endif  //  __CUDA_RHO_P_H__
