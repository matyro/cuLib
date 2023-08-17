
#include <stdio.h>

#include "cuLib/memory.cuh"

#include <catch2/catch_test_macros.hpp>

__global__ void memory_copy(float *a, float *b, unsigned int N)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    if (j >= N)
    {
        return;
    }

    b[j] = a[j];
}



// Individual test subgroups can be put into
// SECTION( "resizing bigger changes size and capacity" )
TEST_CASE("Copy Memory", "[cuda_memory]")
{
    constexpr unsigned int N = 16;

    SECTION("Generate Memory")
    {
        Memory<float> a(N);
        Memory<float> b(N);

        REQUIRE(cudaGetLastError() == cudaSuccess);
    }

    SECTION("Set Memory")
    {
        Memory<float> a(N);
        Memory<float> b(N);

        int i = 0;
        for(auto& a_itr : a)
        {
            a_itr = i++;
        }

        i = 0;
        for(auto& a_itr : a)
        {
            REQUIRE(a_itr == i);
            i++;
        }
    }

    SECTION("Copy Memory")
    {
        Memory<float> a(N);
        Memory<float> b(N);
        int i = 0;
        for(auto& a_itr : a)
        {
            a_itr = i++;
        }

        a.copyToDevice();
        REQUIRE(cudaGetLastError() == cudaSuccess);

        memory_copy<<<1, N>>>(a.getDevicePtr(), b.getDevicePtr(), N);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        cudaDeviceSynchronize();
        REQUIRE(cudaGetLastError() == cudaSuccess);

        b.copyToHost();
        REQUIRE(cudaGetLastError() == cudaSuccess);

        for(int i = 0; i < N; i++)
        {
            REQUIRE(a[i] == b[i]);
        }

    }
    
}