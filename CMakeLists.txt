cmake_minimum_required(VERSION 3.1)
project(OverfitShapes LANGUAGES CXX)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/workspace" CACHE PATH "Install location to move all the files" FORCE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_RENDERER "Build renderer" OFF)
if (BUILD_RENDERER)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 30 35 50 52 53 60 61 62 70 72 75 CACHE STRING "CUDA architectures" FORCE)
endif()


#Build fast winding numbers library
find_package(TBB REQUIRED)

file(GLOB WNFILES ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/WindingNumber/*.cpp)
add_library(FastWindingNumbers STATIC ${WNFILES})

target_include_directories(FastWindingNumbers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/WindingNumbers/)
target_link_libraries(FastWindingNumbers PUBLIC TBB::tbb)

set(3rdparty_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(UNIFORM_SAMPLER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/UniformSampleNBall)
set(UTILITY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Utilities)

#Build point sampler
find_package(Eigen3 REQUIRED)

option(BUILD_PYTHON "Build Python linkage" ON)
if (BUILD_PYTHON)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11")
endif()

option(BUILD_EXAMPLES "Build examples" ON)


add_subdirectory(SDFSampler)

if (BUILD_RENDERER)
    add_subdirectory(RayMarcher)
endif()
