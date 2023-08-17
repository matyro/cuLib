#pragma once

#include <cstddef>



class Context
{
private:
    int cudaDevice_;

    void* deviceMemory1_;
    void* deviceMemory2_;
    void* hostMemory_;

    int currentDeviceMemory_;

    cudaError_t last_error_;

public:
    Context(int device, bool pint=false);
    ~Context();
   
    void synchronize();
};
