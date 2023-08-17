cmake_minimum_required(VERSION 3.16)


project(cuLib
    VERSION 1.0
    LANGUAGES CXX
    )


if(NOT CMAKE_BUILD_TYPE)
message(STATUS "No build type has been specified.")
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        message(WARNING "Build type set to ${CMAKE_BUILD_TYPE}. This build type should only be used for development "
            "and debugging purposes and not for production!")
    else()
        message(STATUS "Build type set to ${CMAKE_BUILD_TYPE}.")
    endif()
endif()

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
    message(STATUS
        "Set -fPIC=True because it's not further specified and neccessary for "
        "the python bindings.")
    SET(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()
  

find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)



set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_TYPE Debug)

#add_compile_options( ) # <- Clashes with nvcc
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)



## Additional Includes

# Activate linker optimizations
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if( supported )
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_link_options(-flto)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

add_subdirectory(externals/Catch2)

## Options
option(BUILD_PYTHON "build python interface" ON)

############################## PYTHON
if(BUILD_PYTHON)
    message("Generation of python bindings enabled!")
    add_compile_definitions(WITH_PYTHON)
    find_package (Python3 COMPONENTS Interpreter Development)

    message("Python_FOUND:${Python_FOUND}")
    message("Python_VERSION:${Python_VERSION}")
    message("Python_Development_FOUND:${Python_Development_FOUND}")
    message("Python_LIBRARIES:${Python_LIBRARIES}")

    include_directories(${PYTHON_INCLUDE_DIRS})
   

    add_subdirectory(externals/pybind11)
endif()

############################## CUDA

include(CheckLanguage)
check_language(CUDA)
message("Cuda Compiler: ${CMAKE_CUDA_COMPILER}")
if( CMAKE_CUDA_COMPILER )
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES 70)

    set(CUDA_AVAILABLE "YES")
    message(STATUS "CUDA is OK")

    set(CMAKE_CUDA_STANDARD 17)        
    
    enable_language(CUDA)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

else ()
    set(CUDA_AVAILABLE "NO")
    message(STATUS "CUDA is NOT OK")
endif()

if( CUDA_AVAILABLE )
    add_subdirectory(cuda)
endif()

############### Include Tests

enable_testing()
add_subdirectory(tests)  