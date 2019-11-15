#ifndef __PARTICLES_UTILS__H__
#define __PARTICLES_UTILS__H__
#include <vector>
#include "../species_advance/species_advance.h"
std::vector<particle_t> get_particles_from_device(
    particle_t* p,
    int n,
    std::vector<particle_mover_t> movers);

#endif // __PARTICLES_UTILS__H__