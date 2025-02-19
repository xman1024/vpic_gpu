/*
 * Written by:
 *   Iwona Kotlarska, Łukasz Kondraciuk
 *   University of Warsaw
 *   2019 - port to CUDA for SC19 student cluster competition
 *
 */
#ifndef __PARTICLES_UTILS__H__
#define __PARTICLES_UTILS__H__
#include <vector>

#include "../species_advance/species_advance.h"
std::vector<particle_t> get_particles_from_device(
    particle_t* p,
    int n,
    const std::vector<particle_mover_t>& movers);

#endif  // __PARTICLES_UTILS__H__
