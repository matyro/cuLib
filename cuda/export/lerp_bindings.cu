#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
namespace py = pybind11;

#include "cuLib/texture.cuh"
#include "cuLib/memory.cuh"
#include "cuLib/context.cuh"

__global__ void lerp_kernel(cudaTextureObject_t tex, float* io, const unsigned int N)
{int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N)
        return;

    const float tex_coords = (float)io[idx] + 0.5;  
    io[idx] = tex2D<float>(tex, tex_coords, 1 );    
}

std::vector<double> execute_lerp(std::vector<double> input, std::vector<float> table, double scale)
{       
    cudaError_t last_error_;

    Memory<float> io(input.size());

    for (int i = 0; i < input.size(); i++)
    {       
        io[i] = input[i];
    }
    io.copyToDevice();

    cudaArray *dArray_;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    last_error_ = cudaMallocArray(&dArray_, &channelDesc, table.size(), 1, cudaArrayDefault);
    if (last_error_ != cudaSuccess)
        throw std::runtime_error("cudaMallocArray failed with " + std::string(cudaGetErrorString(last_error_)));


    last_error_ = cudaMemcpy2DToArray(dArray_, 0, 0, table.data(), table.size() * sizeof(float), table.size() * sizeof(float), 1, cudaMemcpyHostToDevice);
    if (last_error_ != cudaSuccess)
        throw std::runtime_error("cudaMemcpy2DToArray failed with " + std::string(cudaGetErrorString(last_error_)) + " " + std::to_string(last_error_));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dArray_;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType; // Read data as provided type, no casting
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t textureObject_;

    // create texture object: we only have to do this once!
    last_error_ = cudaCreateTextureObject(&textureObject_, &resDesc, &texDesc, NULL);
    if (last_error_ != cudaSuccess)
        throw std::runtime_error("cudaCreateTextureObject failed with " + std::string(cudaGetErrorString(last_error_)));

    lerp_kernel<<<(input.size() + 255) / 256, 256>>>(textureObject_, io.getDevicePtr(), input.size() );

    auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        std::cout << "Kernel failed: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Kernel failed");
    }

    io.copyToHost();

    return std::vector<double>(io.begin(), io.end());
}


/*
std::vector<double> execute_lerp(std::vector<double> input, std::vector<float> table, double scale)
{
    Context cudaContext(0);  
    cudaContext.synchronize(); 

    Memory<float> io(input.size());
    Texture tex(table.size());

    tex.copyToTexture(table.data(), table.size());


    lerp_kernel<<<(input.size() + 255) / 256, 256>>>(tex.getTextureObject(), io.getDevicePtr(), input.size() );

    // Wait until kernel is finished
    auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        std::cout << "Kernel failed: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Kernel failed " + std::string(cudaGetErrorString(error)) + " " + std::to_string(error));
    }
    
  
    io.copyToHost();
  
    std::vector<double> result(input.size());

    std::copy(io.begin(), io.end(), result.begin());
    return result;
}*/

/*
py::array_t<double> execute_lerp_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> tablePy, double scale, py::array_t<double, py::array::c_style | py::array::forcecast> inputPy)
{
    Context cudaContext(0);

    py::buffer_info table = tablePy.request();
    unsigned int Ntable = table.size;

    py::buffer_info input = inputPy.request();
    unsigned int Ninput = input.size;

    std::cout << "Tex Size: " << Ntable << " Querry Size:" << Ninput << std::endl;

    Texture tex(Ntable);
    tex.copyToTexture(reinterpret_cast<float*>(table.ptr), Ntable);



    Memory<float> io(Ninput);
    for (int i = 0; i < Ninput; i++)
    {
        io[i] = reinterpret_cast<double*>(input.ptr)[i];
    }
    io.copyToDevice();

    cudaContext.synchronize();
   
    std::cout << "Kernel: " << std::endl;
    lerp<<<(Ninput + 255) / 256, 256>>>(tex.getTextureObject(), io.getDevicePtr(), scale, Ninput);

    // Wait until kernel is finished
    auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        std::cout << "Kernel failed: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Kernel failed");
    }
    
    std::cout << "Kernel finished" << std::endl;

    //io.copyToHost();

    std::cout << "Copy finished" << std::endl;

    auto result = py::array_t<double>(input.size);
    py::buffer_info output = result.request();

    std::copy(io.begin(), io.end(), reinterpret_cast<double*>(output.ptr));

    std::cout << "result: " << std::endl;
    for(int i = 0; i< Ninput; i++)
    {
        std::cout << reinterpret_cast<double*>(output.ptr)[i] << std::endl;
    }

    return result;
}*/

void bind_lerp(py::module &m)
{
   // m.def("execute_lerp_numpy", &execute_lerp, "execute lerp");
    m.def("execute_lerp", &execute_lerp, "execute lerp");
}