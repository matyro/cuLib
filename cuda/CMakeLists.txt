
message("Work on CUDA CMake Script")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/impl")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")


if(BUILD_PYTHON)
    include_directories(${pybind11_INCLUDE_DIRS})
    add_subdirectory(export)
endif()

add_subdirectory(src)
link_directories(src)

