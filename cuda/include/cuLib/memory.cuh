#pragma once

#include <memory>

// Cuda Memory abstraction
template <class TType>
class Memory
{
private:
    TType* hData_ptr_;
    TType* dData_ptr_;
    unsigned int size_;

    cudaError_t last_error_;

public:
    Memory(unsigned int size);
    ~Memory();

    // Synchronous copy commands
    void copyToDevice();
    void copyToHost();

    TType* getHostPtr();
    TType* getDevicePtr();

    // Iterator functions:
    TType* begin();
    TType* end();
   
    const TType* cbegin() const;
    const TType* cend() const;

    TType& operator[](unsigned int index);
    const TType& operator[](unsigned int index) const;
    
};

#include "cuLib/memory.cu"