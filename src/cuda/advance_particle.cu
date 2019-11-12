#include "../util/util_base.h"
#include "advance_particle.h"
#include "debug.h"
#include "perf_measure.h"
#include "utils.h"

__global__ void particle_move_kernel(particle_t* p0,
                                     interpolator_t* f0,
                                     particle_mover_t* pmovers,
                                     accumulator_t* a0,
                                     int n,
                                     int* moved,
                                     const float qdt_2mc,
                                     const float cdt_dx,
                                     const float cdt_dy,
                                     const float cdt_dz,
                                     const float qsp) {
    const float one            = 1.0;
    const float one_third      = 1.0 / 3.0;
    const float two_fifteenths = 2.0 / 15.0;
    float dx, dy, dz, ux, uy, uz, q;
    float hax, hay, haz, cbx, cby, cbz;
    float v0, v1, v2, v3, v4, v5;
    float* a;
    int ii;

    particle_mover_t local_pm;

    int i            = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        particle_t* p = p0 + i;
        dx            = p->dx;  // Load position
        dy            = p->dy;
        dz            = p->dz;
        ii            = p->i;

        const interpolator_t* f = f0 + ii;  // Interpolate E

        hax = qdt_2mc *
              ((f->ex + dy * f->dexdy) + dz * (f->dexdz + dy * f->d2exdydz));

        hay = qdt_2mc *
              ((f->ey + dz * f->deydz) + dx * (f->deydx + dz * f->d2eydzdx));

        haz = qdt_2mc *
              ((f->ez + dx * f->dezdx) + dy * (f->dezdy + dx * f->d2ezdxdy));

        cbx = f->cbx + dx * f->dcbxdx;  // Interpolate B
        cby = f->cby + dy * f->dcbydy;
        cbz = f->cbz + dz * f->dcbzdz;

        ux = p->ux;  // Load momentum
        uy = p->uy;
        uz = p->uz;
        q  = p->w;

        ux += hax;  // Half advance E
        uy += hay;
        uz += haz;

        v0 = qdt_2mc / sqrt(one + (ux * ux + (uy * uy + uz * uz)));

        // Boris - scalars
        v1 = cbx * cbx + (cby * cby + cbz * cbz);
        v2 = (v0 * v0) * v1;
        v3 = v0 * (one + v2 * (one_third + v2 * two_fifteenths));
        v4 = v3 / (one + v1 * (v3 * v3));
        v4 += v4;

        v0 = ux + v3 * (uy * cbz - uz * cby);  // Boris - uprime
        v1 = uy + v3 * (uz * cbx - ux * cbz);
        v2 = uz + v3 * (ux * cby - uy * cbx);

        ux += v4 * (v1 * cbz - v2 * cby);  // Boris - rotation
        uy += v4 * (v2 * cbx - v0 * cbz);
        uz += v4 * (v0 * cby - v1 * cbx);

        ux += hax;  // Half advance E
        uy += hay;
        uz += haz;

        p->ux = ux;  // Store momentum
        p->uy = uy;
        p->uz = uz;

        v0 = one / sqrt(one + (ux * ux + (uy * uy + uz * uz)));
        // Get norm displacement

        ux *= cdt_dx;
        uy *= cdt_dy;
        uz *= cdt_dz;

        ux *= v0;
        uy *= v0;
        uz *= v0;

        v0 = dx + ux;  // Streak midpoint (inbnds)
        v1 = dy + uy;
        v2 = dz + uz;

        v3 = v0 + ux;  // New position
        v4 = v1 + uy;
        v5 = v2 + uz;

        // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
        if (v3 <= one && v4 <= one && v5 <= one &&  // Check if inbnds
            -v3 <= one && -v4 <= one && -v5 <= one) {
            // Common case (inbnds).  Note: accumulator values are 4 times
            // the total physical charge that passed through the appropriate
            // current quadrant in a time-step.

            q *= qsp;

            p->dx = v3;  // Store new position
            p->dy = v4;
            p->dz = v5;

            dx = v0;  // Streak midpoint
            dy = v1;
            dz = v2;

            v5 = q * ux * uy * uz * one_third;  // Compute correction

            a = (float*)(a0 + ii);  // Get accumulator

#define ACCUMULATE_J(X, Y, Z, offset)                           \
    v4 = q * u##X;   /* v2 = q ux                            */ \
    v1 = v4 * d##Y;  /* v1 = q ux dy                         */ \
    v0 = v4 - v1;    /* v0 = q ux (1-dy)                     */ \
    v1 += v4;        /* v1 = q ux (1+dy)                     */ \
    v4 = one + d##Z; /* v4 = 1+dz                            */ \
    v2 = v0 * v4;    /* v2 = q ux (1-dy)(1+dz)               */ \
    v3 = v1 * v4;    /* v3 = q ux (1+dy)(1+dz)               */ \
    v4 = one - d##Z; /* v4 = 1-dz                            */ \
    v0 *= v4;        /* v0 = q ux (1-dy)(1-dz)               */ \
    v1 *= v4;        /* v1 = q ux (1+dy)(1-dz)               */ \
    v0 += v5;        /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */ \
    v1 -= v5;        /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */ \
    v2 -= v5;        /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */ \
    v3 += v5;        /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */ \
    atomicAdd(&a[offset + 0], v0);                              \
    atomicAdd(&a[offset + 1], v1);                              \
    atomicAdd(&a[offset + 2], v2);                              \
    atomicAdd(&a[offset + 3], v3);
            // tutaj będzie potrzebna redukcyjka
            ACCUMULATE_J(x, y, z, 0);
            ACCUMULATE_J(y, z, x, 4);
            ACCUMULATE_J(z, x, y, 8);

#undef ACCUMULATE_J
        }

        else  // Unlikely
        {
            local_pm.dispx    = ux;
            local_pm.dispy    = uy;
            local_pm.dispz    = uz;
            local_pm.i        = p - p0;
            int save_idx      = atomicAdd(moved, 1);
            pmovers[save_idx] = local_pm;
        }
    }
}

void run_kernel(
                particle_t* device_p0,// wielkość n
                int n,
                const interpolator_t* f0,  // wielkość accumulator_size
                accumulator_t* a0,         // wielkość interpolator_size
                particle_mover_t* pm,
                particle_mover_t* device_pm,
                const grid_t* g,
                const float qdt_2mc,
                const float cdt_dx,
                const float cdt_dy,
                const float cdt_dz,
                const float qsp,
                const int max_nm,
                int* nm,
                int* skipped) {
    // Rozmiary kopiowanych tablic
    const int grid_size         = (g->nx + 2) * (g->ny + 2) * (g->nz + 2);
    const int accumulator_size  = POW2_CEIL(grid_size, 2);
    const int interpolator_size = grid_size;

    int moved = 0;

    interpolator_t* device_f0;
    accumulator_t* device_a0;
    particle_mover_t* device_pmovers;
    int64_t* device_neighbours;
    int* device_moved;
    int* device_moved_2;

    // Alokacja
    CUDA_CHECK(cudaMalloc((void**)&device_f0,
                          sizeof(interpolator_t) * interpolator_size));
    CUDA_CHECK(cudaMalloc((void**)&device_a0,
                          sizeof(accumulator_t) * accumulator_size));

    CUDA_CHECK(
        cudaMalloc((void**)&device_pmovers, sizeof(particle_mover_t) * n));
    CUDA_CHECK(cudaMalloc((void**)&device_neighbours,
                          sizeof(int64_t) * grid_size * 6));
    CUDA_CHECK(cudaMalloc((void**)&device_moved, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&device_moved_2, sizeof(int)));

    // Kopiowanie tam
    CUDA_CHECK(cudaMemcpy(device_f0, f0,
                          sizeof(interpolator_t) * interpolator_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_a0, a0,
                          sizeof(accumulator_t) * accumulator_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_neighbours, g->neighbor,
                          sizeof(int64_t) * grid_size * 6,
                          cudaMemcpyHostToDevice));
    device_set_var(device_moved, 0);
    device_set_var(device_moved_2, 0);

    // wywołanie kernela TODO liczba bloków powinna z grubsza odpowiadać liczbie
    // SMów
    particle_move_kernel<<<1024, 1024>>>(device_p0, device_f0, device_pmovers,
                                         device_a0, n, device_moved, qdt_2mc,
                                         cdt_dx, cdt_dy, cdt_dz, qsp);

    moved = device_fetch_var(device_moved);

    cuda_move_p(device_p0, device_pmovers, moved, device_a0, device_neighbours,
                qsp, device_moved_2, device_pm, g->rangeh, g->rangel);
    // kopiowanie z powrotem
    CUDA_CHECK(cudaMemcpy(a0, device_a0,
                          sizeof(accumulator_t) * accumulator_size,
                          cudaMemcpyDeviceToHost));
    *nm = device_fetch_var(device_moved_2);
    CUDA_CHECK(cudaMemcpy(pm, device_pm, sizeof(particle_mover_t) * (*nm),
                          cudaMemcpyDeviceToHost));
    // Free
    CUDA_CHECK(cudaFree(device_f0));
    CUDA_CHECK(cudaFree(device_a0));
    CUDA_CHECK(cudaFree(device_pmovers));
    CUDA_CHECK(cudaFree(device_neighbours));
    CUDA_CHECK(cudaFree(device_moved));
    CUDA_CHECK(cudaFree(device_moved_2));
}
