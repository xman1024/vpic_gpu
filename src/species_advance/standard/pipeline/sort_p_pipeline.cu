#include "thrust/sort.h"
#define IN_spa

#include "../../../cuda/utils.h"
#include "../../../util/pipelines/pipelines_exec.h"
#include "spa_private.h"


struct particle_compare
{
    __host__ __device__ bool operator()(const particle_t& x, const particle_t& y)
    {
        return x.i < y.i;
    }
};

void sort_p_pipeline(species_t* sp) {
    if (!sp) {
        ERROR(("Bad args"));
    }

    sp->last_sorted = sp->g->step;

    particle_t* p = sp->device_p0;
    int np = sp->np;

    thrust::sort(thrust::device, p, p + np, particle_compare());
}
