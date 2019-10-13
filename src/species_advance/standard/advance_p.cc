#define IN_spa

#include "../species_advance.h"
#include "../../cuda/advance_particle.h"
#include "../../cuda/debug.h"
#include <iostream>
#include <iomanip>
//----------------------------------------------------------------------------//
// Top level function to select and call particle advance function using the
// desired particle advance abstraction.  Currently, the only abstraction
// available is the pipeline abstraction.
//----------------------------------------------------------------------------//

void
advance_p( species_t * RESTRICT sp,
           accumulator_array_t * RESTRICT aa,
           const interpolator_array_t * RESTRICT ia )
{
  species_t sp2 = *sp;
  DEBUG(aa->stride)
  
  sp2.p = new particle_t[sp2.np];
  for (int i = 0; i < sp2.np; ++i)
    sp2.p[i] = sp->p[i];
  advance_p_cuda( &sp2, aa, ia );
  std::cerr << "OK" << std::endl;
  advance_p_pipeline( sp, aa, ia );
  double tot_diff = 0.;
  DEBUG(sp2.np)
  for (int i = 0; i < sp2.np; ++i)
    tot_diff += diff(sp2.p[i], sp->p[i]);
  std::cerr << std::setprecision(13);
  DEBUG(tot_diff)
  double avg_diff = tot_diff / sp2.np;
  DEBUG(avg_diff)
}
