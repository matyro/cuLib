
#include <iostream>

#include "cuLib/context.cuh"

// Cuda Includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

Context::Context(int device, bool print)
    : cudaDevice_(device)
{
    int nDevices;
    last_error_ = cudaGetDeviceCount(&nDevices);
    if (last_error_ != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDeviceCount failed with " + std::string(cudaGetErrorString(last_error_)));
    }

    if (nDevices < cudaDevice_ + 1)
    {
        throw std::runtime_error("cuda device ID " + std::to_string(cudaDevice_) + " does not exist!");
    }

    if (print)
    {
        for (int i = 0; i < nDevices; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            std::cout << "Device Number: " << cudaDevice_ << std::endl;
            std::cout << "  Device name: " << prop.name << std::endl;
            std::cout << "  Memory Clock Rate (MHz): " << prop.memoryClockRate / 1024 << std::endl;
            std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
            std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
            std::cout << "  Total global memory (Gbytes) " << (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0 << std::endl;
            std::cout << "  Shared memory per block (Kbytes) " << (float)(prop.sharedMemPerBlock) / 1024.0 << std::endl;
            std::cout << "  minor-major: " << prop.minor << "-" << prop.major << std::endl;
            std::cout << "  Warp-size: " << prop.warpSize << std::endl;
            std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "yes" : "no") << std::endl;
            std::cout << "  Concurrent computation/communication: " << (prop.deviceOverlap ? "yes" : "no") << std::endl;
            std::cout << std::endl;
        }
    }

    // Initialize the CUDA runtime.
    last_error_ = cudaSetDevice(cudaDevice_);
    if (last_error_ != cudaSuccess)
    {
        throw std::runtime_error("cudaSetDevice failed with " + std::string(cudaGetErrorString(last_error_)));
    }
}

Context::~Context()
{
    // Shutdown the CUDA runtime.
    last_error_ = cudaDeviceReset();
    if (last_error_ != cudaSuccess)
    {
        throw std::runtime_error("cudaDeviceReset failed with" + std::string(cudaGetErrorString(last_error_)));
    }
}

void Context::synchronize()
{
    last_error_ = cudaDeviceSynchronize();
    if (last_error_ != cudaSuccess)
    {
        throw std::runtime_error("cudaDeviceSynchronize returned error " + std::string(cudaGetErrorString(last_error_)) + " after launching kernel!");
    }
}