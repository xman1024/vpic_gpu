#include "../util/util_base.h"
#include "advance_particle.h"
#include "debug.h"
#include "perf_measure.h"
#include "utils.h"

__global__ void cuda_move_p_kernel(particle_t* p0,
                                   particle_mover_t* pm,
                                   int n,
                                   accumulator_t* a0,
                                   int64_t* neighbours,
                                   const float qsp,
                                   int* nm,
                                   particle_mover_t* pm_save,
                                   int64_t rangeh,
                                   int64_t rangel) {
    float s_midx, s_midy, s_midz;
    float s_dispx, s_dispy, s_dispz;
    float s_dir[3];
    float v0, v1, v2, v3, v4, v5, q;
    int axis, face;
    int64_t neighbor;
    float* a;

    int idx          = 0;
    const int stride = 1;

    for (; idx < n; idx += stride) {
        particle_t* p = p0 + pm[idx].i;

        q = qsp * p->w;

        for (;;) {
            s_midx = p->dx;
            s_midy = p->dy;
            s_midz = p->dz;

            s_dispx = pm[idx].dispx;
            s_dispy = pm[idx].dispy;
            s_dispz = pm[idx].dispz;

            s_dir[0] = (s_dispx > 0.0f) ? 1.0f : -1.0f;
            s_dir[1] = (s_dispy > 0.0f) ? 1.0f : -1.0f;
            s_dir[2] = (s_dispz > 0.0f) ? 1.0f : -1.0f;

            // Compute the twice the fractional distance to each potential
            // streak/cell face intersection.
            v0 = (s_dispx == 0.0f) ? 3.4e38f : (s_dir[0] - s_midx) / s_dispx;
            v1 = (s_dispy == 0.0f) ? 3.4e38f : (s_dir[1] - s_midy) / s_dispy;
            v2 = (s_dispz == 0.0f) ? 3.4e38f : (s_dir[2] - s_midz) / s_dispz;

            // Determine the fractional length and axis of current streak. The
            // streak ends on either the first face intersected by the
            // particle track or at the end of the particle track.
            //
            //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
            //   axis 3        ... streak ends at end of the particle track
            /**/ v3 = 2.0f, axis = 3;
            if (v0 < v3)
                v3 = v0, axis = 0;
            if (v1 < v3)
                v3 = v1, axis = 1;
            if (v2 < v3)
                v3 = v2, axis = 2;
            v3 *= 0.5f;

            // Compute the midpoint and the normalized displacement of the streak
            s_dispx *= v3;
            s_dispy *= v3;
            s_dispz *= v3;
            s_midx += s_dispx;
            s_midy += s_dispy;
            s_midz += s_dispz;

            // Accumulate the streak.  Note: accumulator values are 4 times
            // the total physical charge that passed through the appropriate
            // current quadrant in a time-step
            v5 = q * s_dispx * s_dispy * s_dispz * (1. / 3.);
            a  = (float*)(a0 + p->i);
#define accumulate_j(X, Y, Z)                                      \
    v4 = q * s_disp##X; /* v2 = q ux                            */ \
    v1 = v4 * s_mid##Y; /* v1 = q ux dy                         */ \
    v0 = v4 - v1;       /* v0 = q ux (1-dy)                     */ \
    v1 += v4;           /* v1 = q ux (1+dy)                     */ \
    v4 = 1 + s_mid##Z;  /* v4 = 1+dz                            */ \
    v2 = v0 * v4;       /* v2 = q ux (1-dy)(1+dz)               */ \
    v3 = v1 * v4;       /* v3 = q ux (1+dy)(1+dz)               */ \
    v4 = 1 - s_mid##Z;  /* v4 = 1-dz                            */ \
    v0 *= v4;           /* v0 = q ux (1-dy)(1-dz)               */ \
    v1 *= v4;           /* v1 = q ux (1+dy)(1-dz)               */ \
    v0 += v5;           /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */ \
    v1 -= v5;           /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */ \
    v2 -= v5;           /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */ \
    v3 += v5;           /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */ \
    a[0] += v0;                                                    \
    a[1] += v1;                                                    \
    a[2] += v2;                                                    \
    a[3] += v3
            accumulate_j(x, y, z);
            a += 4;
            accumulate_j(y, z, x);
            a += 4;
            accumulate_j(z, x, y);
#undef accumulate_j

            // Compute the remaining particle displacment
            pm[idx].dispx -= s_dispx;
            pm[idx].dispy -= s_dispy;
            pm[idx].dispz -= s_dispz;

            // Compute the new particle offset
            p->dx += s_dispx + s_dispx;
            p->dy += s_dispy + s_dispy;
            p->dz += s_dispz + s_dispz;

            // If an end streak, return success (should be ~50% of the time)

            if (axis == 3)
                break;

            // Determine if the particle crossed into a local cell or if it
            // hit a boundary and convert the coordinate system accordingly.
            // Note: Crossing into a local cell should happen ~50% of the
            // time; hitting a boundary is usually a rare event.  Note: the
            // entry / exit coordinate for the particle is guaranteed to be
            // +/-1 _exactly_ for the particle.

            v0               = s_dir[axis];
            (&(p->dx))[axis] = v0;  // Avoid roundoff fiascos--put the particle
                                    // _exactly_ on the boundary.
            face = axis;
            if (v0 > 0)
                face += 3;
            neighbor = neighbours[6 * p->i + face];

            if (UNLIKELY(neighbor == reflect_particles)) {
                // Hit a reflecting boundary condition.  Reflect the particle
                // momentum and remaining displacement and keep moving the
                // particle.
                (&(p->ux))[axis]         = -(&(p->ux))[axis];
                (&(pm[idx].dispx))[axis] = -(&(pm[idx].dispx))[axis];
                continue;
            }

            if (UNLIKELY(neighbor < rangel || neighbor > rangeh)) {
                // Cannot handle the boundary condition here.  Save the updated
                // particle position, face it hit and update the remaining
                // displacement in the particle mover.
                p->i             = 8 * p->i + face;
                pm_save[(*nm)++] = pm[idx];
                break;
            }

            // Crossed into a normal voxel.  Update the voxel index, convert the
            // particle coordinate system and keep moving the particle.

            p->i = neighbor - rangel;  // Compute local index of neighbor
            /**/                          // Note: neighbor - g->rangel < 2^31 / 6
            (&(p->dx))[axis] = -v0;       // Convert coordinate system
        }
    }
}

void cuda_move_p(particle_t* p0,
                 particle_mover_t* pm,
                 int n,
                 accumulator_t* a0,
                 int64_t* neighbours,
                 const float qsp,
                 int* nm,
                 particle_mover_t* pm_save,
                 int64_t rangeh,
                 int64_t rangel) {
    cuda_move_p_kernel<<<1, 1>>>(p0, pm, n, a0, neighbours, qsp, nm, pm_save, rangeh,
                                 rangel);
}
