#include <vector>
#include "particles_utils.h"
#include "utils.h"
#include "../util/util_base.h"
#include "../species_advance/species_advance.h"

__global__ void get_particles_kernel(particle_t* p,
                                     particle_mover_t* movers,
                                     int n,
                                     int pm,
                                     particle_t* res) {
    int j = 0;
    for (int i = pm - 1; i >= 0; --i, ++j) {
        int ind = movers[i].i;
        res[j]  = p[ind];
        p[ind]  = p[n - 1 - j];
    }
}

std::vector<particle_t> get_particles_from_device(
    particle_t* p,
    int n,
    std::vector<particle_mover_t> movers) {
    int pm = movers.size();
    if (pm == 0)
        return {};
    std::vector<particle_t> res(pm);
    particle_mover_t* device_movers;
    CUDA_CHECK(
        cudaMalloc((void**)&device_movers, sizeof(particle_mover_t) * pm));
    CUDA_CHECK(cudaMemcpy(device_movers, movers.data(),
                          sizeof(particle_mover_t) * pm,
                          cudaMemcpyHostToDevice));
    particle_t* device_res;
    CUDA_CHECK(cudaMalloc((void**)&device_res, sizeof(particle_t) * pm));
    get_particles_kernel<<<1, 1>>>(p, device_movers, n, pm, device_res);
    CUDA_CHECK(cudaMemcpy(res.data(), device_res, sizeof(particle_t) * pm,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_res));
    CUDA_CHECK(cudaFree(device_movers));
    
    res = std::vector<particle_t>(res.rbegin(), res.rend());

    return res;
}