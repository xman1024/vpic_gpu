#ifndef __CUDA_UTILS__H__
#define __CUDA_UTILS__H__

#include <cuda.h>
#include <iostream>


void detect_cuda_init();

#define CUDA_CHECK(error)                                                    \
    {                                                                        \
        auto status = static_cast<cudaError_t>(error);                       \
        if (status != cudaSuccess) {                                         \
            std::cerr << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                      << ", CUDA error:" << cudaGetErrorString(status);      \
            exit(1);                                                         \
        }                                                                    \
    }

#endif // __CUDA_UTILS__H__