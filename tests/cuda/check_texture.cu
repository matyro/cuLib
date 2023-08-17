#include "cuLib/context.cuh"
#include "cuLib/texture.cuh"
#include "cuLib/memory.cuh"

#include <catch2/catch_all.hpp>

#include <iostream>

#include <stdio.h>

__global__ void lerp_texture(cudaTextureObject_t tex, float *io, const unsigned int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N)
        return;

    const float tex_coords = (float)idx + 0.5;

    printf("tex_coords: %f\n", tex_coords);
    printf("tex1D<float>(tex, tex_coords): %f\n", tex1D<float>(tex, tex_coords));

    io[idx] = tex2D<float>(tex, tex_coords, 1);
}


TEST_CASE("Test Small", "[cuda_texture]")
{
    Context cudaContext(0);
    constexpr unsigned int N = 16;

    Memory<float> io(N);

    float src[N];
    for (int i = 0; i < N; i++)
    {
        src[i] = float(i);
    }

    Texture tex(N);
    tex.copyToTexture(src, N);
    

    std::cout << "Sync before kernel" << std::endl;
    // cudaContext.synchronize();

    std::cout << "Running kernel" << std::endl;
    lerp_texture<<<(N + 255) / 256, 256>>>(tex.getTextureObject(), io.getDevicePtr(), N);

   if(cudaGetLastError() != cudaSuccess)
    {
        std::cout << "Kernel failed with " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }   

    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        std::cout << "Sync after failed with " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    // cudaContext.synchronize();

   
    io.copyToHost();

    for (int i = 0; i < N; i++)
    {
        if(io[i] != src[i])
        {
            std::cout << "Error at " << i << " " << io[i] << " != " << src[i] << std::endl;
        }
    }

    auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }   
}

TEST_CASE("Test Med", "[cuda_texture]")
{
    std::cout << "TestPrecision" << std::endl;

    Context cudaContext(0);

    constexpr unsigned int N = 4096;

    Memory<float> io(N);


    float src[N];
    for (int i = 0; i < N; i++)
    {
        src[i] = float(i);
    }

    SECTION("Generate Texture")
    {
        Texture tex(N);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        tex.copyToTexture(src, N);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        std::cout << "Running kernel" << std::endl;
        lerp_texture<<<(N + 255)/256, 256>>>(tex.getTextureObject(), io.getDevicePtr(), N);

        REQUIRE(cudaGetLastError() == cudaSuccess);

        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        // Check for errors:
        REQUIRE(cudaGetLastError() == cudaSuccess);

        io.copyToHost();

        for(int i=0; i<N; i++)
        {
            REQUIRE(io[i] == Catch::Approx(src[i]));
        }

        auto error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
    }
}


TEST_CASE("Test Big", "[cuda_texture]")
{
    std::cout << "TestPrecision" << std::endl;

    Context cudaContext(0, true);

    constexpr unsigned int N = 12000;

    Memory<float> io(N);


    float src[N];
    for (int i = 0; i < N; i++)
    {
        src[i] = float(i);
    }

    SECTION("Generate Texture")
    {
        Texture tex(N);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        tex.copyToTexture(src, N);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        std::cout << "Running kernel" << std::endl;
        lerp_texture<<<(N + 255)/256, 256>>>(tex.getTextureObject(), io.getDevicePtr(), N);

        REQUIRE(cudaGetLastError() == cudaSuccess);

        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        // Check for errors:
        REQUIRE(cudaGetLastError() == cudaSuccess);

        io.copyToHost();

        for(int i=0; i<N; i++)
        {
            REQUIRE(io[i] == Catch::Approx(src[i]));
        }

        auto error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
    }
}