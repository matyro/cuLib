
add_library(CudaLib STATIC)

target_compile_options(CudaLib PRIVATE -Wall) # -Wextra -Wpedantic

target_include_directories(CudaLib PUBLIC include "${CMAKE_SOURCE_DIR}/cuda/impl")
target_include_directories(CudaLib PUBLIC include "${CMAKE_SOURCE_DIR}/cuda/include")

# If this is switched on link error:
# /usr/local/cuda/bin/crt/link.stub:87:30: error:  redefinition of ‘const unsigned char def_module_id_str
#set_property(TARGET CudaLib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(cuLib)



