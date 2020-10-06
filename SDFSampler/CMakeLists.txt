add_library(SDFSampler STATIC
    src/PointSampler.cpp
    src/Octree.cpp
    src/MeshLoader.cpp)

MESSAGE(${UNIFORM_SAMPLER_DIR})
message(${3rdparty_DIR})

target_include_directories(SDFSampler PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${EIGEN3_INCLUDE_DIR}
    ${UNIFORM_SAMPLER_DIR}
    ${3rdparty_DIR})

target_link_libraries(SDFSampler PUBLIC
    FastWindingNumbers)

if (NOT WIN32)
    target_link_libraries(SDFSampler PUBLIC stdc++fs)
endif()

if (BUILD_PYTHON)
    add_subdirectory(python)
endif()

