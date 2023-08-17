
#include <memory>

#include <stdexcept>

#include <iostream>

#include "cuLib/texture.cuh"

__host__ Texture::Texture(const size_t N) : width_(N)
{   

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    last_error_ = cudaMallocArray(&dArray_, &channelDesc, sizeof(float) * width_, 1, cudaArrayDefault);
    if (last_error_ != cudaSuccess)
        throw std::runtime_error("Texture::Texture: cudaMallocArray failed width " + std::string(cudaGetErrorString(last_error_)));
}

__host__ Texture::~Texture()
{
    cudaFreeArray(dArray_);

    if (textureObject_ != 0)
        cudaDestroyTextureObject(textureObject_);
}

__host__ void Texture::copyToTexture(const float *const data, const unsigned int N)
{
    if (N != width_)
        throw std::runtime_error("Texture::copyToTexture: data size does not match texture size");

    // (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    last_error_ = cudaMemcpy2DToArray(dArray_, 0, 0, data, N * sizeof(float), N * sizeof(float), 1, cudaMemcpyHostToDevice);
    //last_error_ = cudaMemcpyToArray(dArray_, 0, 0, data, width_ * sizeof(float), cudaMemcpyHostToDevice);
    if (last_error_ != cudaSuccess)
        throw std::runtime_error("Texture::copyToTexture: cudaMemcpyToArray failed width " + std::string(cudaGetErrorString(last_error_)));
}

__host__ cudaTextureObject_t Texture::getTextureObject()
{
    if (textureObject_ == 0)
    {
        // create texture object
        memset(&resDesc_, 0, sizeof(resDesc_));
        resDesc_.resType = cudaResourceTypeArray;
        resDesc_.res.array.array = dArray_;

        memset(&texDesc_, 0, sizeof(texDesc_));
        texDesc_.readMode = cudaReadModeElementType; // Read data as provided type, no casting
        texDesc_.filterMode = cudaFilterModeLinear;
        texDesc_.addressMode[0] = cudaAddressModeWrap;
        texDesc_.normalizedCoords = 0;

        cudaResourceViewDesc viewDesc;
        memset(&viewDesc, 0, sizeof(viewDesc));
        viewDesc.format = cudaResViewFormatFloat1;
        viewDesc.width = width_;
        viewDesc.height = 1;

        // create texture object: we only have to do this once!
        last_error_ = cudaCreateTextureObject(&textureObject_, &resDesc_, &texDesc_, NULL);
        if (last_error_ != cudaSuccess)
            throw std::runtime_error("Texture::getTextureObject: cudaCreateTextureObject failed with " + std::string(cudaGetErrorString(last_error_)));
    }
    return textureObject_;
}
