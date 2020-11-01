if (WIN32)
set(SDFSampler_BUILD_MODE SHARED)
else()
set(SDFSampler_BUILD_MODE STATIC)
endif()

find_package(CGAL REQUIRED)

add_library(Sampler ${SDFSampler_BUILD_MODE}
    src/PointSampler.cpp
    src/MeshLoader.cpp)

target_include_directories(Sampler PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${EIGEN3_INCLUDE_DIR}
    ${CGAL_INCLUDE_DIR}
    ${UNIFORM_SAMPLER_DIR}
    ${3rdparty_DIR})

target_link_libraries(Sampler PUBLIC
    FastWindingNumbers
    ${CGAL_LIBRARIES})

if (NOT WIN32)
    target_link_libraries(Sampler PUBLIC stdc++fs)
endif()

if (BUILD_EXAMPLE)
    add_executable(SamplerExample example/main.cpp)

    target_link_libraries(SamplerExample PUBLIC Sampler)
endif()

if (BUILD_PYTHON)
    pybind11_add_module(pySDFSampler SHARED python/python.cpp)

    target_link_libraries(pySDFSampler PUBLIC Sampler)

    set_target_properties(pySDFSampler PROPERTIES OUTPUT_NAME SDFSampler)

    add_dependencies(pySDFSampler Sampler)
endif()

