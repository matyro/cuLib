ADD_EXECUTABLE( check_cudaLib "check_texture.cu" "check_context.cu"  "check_memory.cu" "check_main.cu")

target_link_libraries(check_cudaLib PRIVATE Catch2::Catch2)
target_link_libraries(check_cudaLib PUBLIC CudaLib)

set_property(TARGET check_cudaLib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME check_cudaLib COMMAND check_cudaLib)

