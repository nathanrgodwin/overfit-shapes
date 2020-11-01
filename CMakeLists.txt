cmake_minimum_required(VERSION 3.1)
project(SDFSampler)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#Build fast winding numbers library
find_package(TBB REQUIRED)

file(GLOB WNFILES ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/WindingNumber/*.cpp)
add_library(FastWindingNumbers STATIC ${WNFILES})

target_include_directories(FastWindingNumbers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/WindingNumbers/)
target_link_libraries(FastWindingNumbers PUBLIC TBB::tbb)

set(3rdparty_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(UNIFORM_SAMPLER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/UniformSampleNBall)

#Build point sampler
find_package(Eigen3 REQUIRED)

option(BUILD_PYTHON "Build Python linkage" ON)
if (BUILD_PYTHON)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11")
endif()

option(BUILD_EXAMPLE "Build sampler example" ON)

add_subdirectory(SDFSampler)
