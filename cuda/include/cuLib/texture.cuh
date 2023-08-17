#pragma once

#include <memory>
#include <stdexcept>
#include <iostream>

class Texture
{
private:
    cudaArray *dArray_;

    const size_t width_;

    cudaTextureObject_t textureObject_;
    cudaChannelFormatDesc channelDesc_;

    cudaResourceDesc resDesc_;
    cudaTextureDesc texDesc_;

    cudaError_t last_error_;

public:
    __host__ Texture(const size_t N);


    __host__ ~Texture();

    __host__ void copyToTexture(const float *const data, const unsigned int N);

    __host__ cudaTextureObject_t getTextureObject();
};
