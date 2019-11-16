#include <cuda.h>
#include <cuda_runtime_api.h>
#include "utils.h"
#include <mpi.h>

#include <iostream>

void set_proper_device() {
    int n_of_devices;
    CUDA_CHECK(cudaGetDeviceCount(&n_of_devices));
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int my_device = world_rank % n_of_devices;
    CUDA_CHECK(cudaSetDevice(my_device));
}

void detect_cuda_init() {
    cudaError_t cuda_status = cudaFree(0);
    if (cuda_status != cudaSuccess) {
        std::cout << "cudaFree(0) failed on DetectCudaInit with error code ("
                  << cuda_status << "): " << cudaGetErrorString(cuda_status)
                  << std::endl;
        std::cout << "CUDA initialization failed - cudaFree(0) "
                     "is first cuda call "
                  << std::endl;
        exit(1);
    }

    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        std::cout
            << "cudaSetDevice(0) failed on DetectCudaInit with error code ("
            << cuda_status << "): " << cudaGetErrorString(cuda_status)
            << std::endl;
        std::cout << "CUDA initialization failed - "
                     "cudaSetDevice(0) is second cuda "
                  << std::endl;
        exit(1);
    }
    
}