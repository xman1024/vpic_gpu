#define IN_sfa

#include "sfa_private.h"

//----------------------------------------------------------------------------//
// Top level function to select and call the proper advance_e function.
//----------------------------------------------------------------------------//

void advance_e(field_array_t* RESTRICT fa, float frac) {
    if (!fa) {
        ERROR(("Bad args"));
    }

    if (frac != 1) {
        ERROR(("standard advance_e does not support frac != 1 yet"));
    }

    // Conditionally execute this when more abstractions are available.
    advance_e_pipeline(fa, frac);
}
