

#include <iostream>
#include <stdexcept>

template <class TType>
Memory<TType>::Memory(unsigned int size) : hData_ptr_(nullptr), dData_ptr_(nullptr), size_(size)
{

    last_error_= cudaMallocHost(&hData_ptr_, size_ * sizeof(TType));
    if (last_error_!= cudaSuccess)
    {
        std::cout << "error allocating pinned host memory: " << cudaGetErrorString(last_error_) << std::endl;
        exit(-1);
    }

    last_error_= cudaMalloc(&dData_ptr_, size_ * sizeof(TType));
    if (last_error_!= cudaSuccess)
    {
        std::cout << "error allocating device memory: " << cudaGetErrorString(last_error_) << std::endl;
        exit(-1);
    }
}

template <class TType>
Memory<TType>::~Memory()
{
    cudaFreeHost(hData_ptr_);
    cudaFree(dData_ptr_);   
}

template <class TType>
void Memory<TType>::copyToDevice()
{
    last_error_= cudaMemcpy(dData_ptr_, hData_ptr_, size_ * sizeof(TType), cudaMemcpyHostToDevice);
    if (last_error_!= cudaSuccess)
    {
        std::cout << "error copying data to device: " << cudaGetErrorString(last_error_) << std::endl;
        exit(-1);
    }
}

template <class TType>
void Memory<TType>::copyToHost()
{  
    last_error_= cudaMemcpy(hData_ptr_, dData_ptr_, size_ * sizeof(TType), cudaMemcpyDeviceToHost);
    if (last_error_!= cudaSuccess)
    {
        std::cout << "error copying data to host: " << cudaGetErrorString(last_error_) << std::endl;
        exit(-1);
    }
}

template <class TType>
TType* Memory<TType>::getHostPtr()
{
    return hData_ptr_;
}

template <class TType>
TType* Memory<TType>::getDevicePtr()
{
    return dData_ptr_;
}

template <class TType>
TType *Memory<TType>::begin()
{
    return hData_ptr_;
}

template <class TType>
TType *Memory<TType>::end()
{
    return hData_ptr_ + size_;
}

template <class TType>
const TType *Memory<TType>::cbegin() const
{
    return hData_ptr_;
}

template <class TType>
const TType *Memory<TType>::cend() const
{
    return hData_ptr_ + size_;
}

template <class TType>
TType &Memory<TType>::operator[](unsigned int index)
{
    if (index >= size_)
        throw std::out_of_range("Index out of range");

    return hData_ptr_[index];
}

template <class TType>
const TType &Memory<TType>::operator[](unsigned int index) const
{
    if (index >= size_)
        throw std::out_of_range("Index out of range");

    return hData_ptr_[index];
}