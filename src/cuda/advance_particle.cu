#include "advance_particle.h"
#include <iostream>

void advance_p_cuda( species_t * RESTRICT sp,
    accumulator_array_t * RESTRICT aa,
    const interpolator_array_t * RESTRICT ia ) {
        std::cerr << "Hello from cuda" << std::endl;

    }