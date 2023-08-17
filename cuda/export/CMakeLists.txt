
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

pybind11_add_module(cuda_python SHARED "cuda_bindings.cu" "lerp_bindings.cu")


target_include_directories(cuda_python PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${PYTHON_INCLUDE_DIRS}>
)

target_link_libraries(cuda_python PUBLIC CudaLib)
target_link_libraries(cuda_python PRIVATE ${PYTHON_LIBRARIES})

set_property(TARGET cuda_python PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS cuda_python EXPORT cuda_pythonTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})



target_sources(cuda_python PUBLIC "lerp_bindings.cu")