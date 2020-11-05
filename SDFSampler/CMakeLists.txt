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

    install(TARGETS 
        SamplerExample
        LIBRARY DESTINATION ./
        RUNTIME DESTINATION ./)

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/cube.obj"
            DESTINATION "${CMAKE_INSTALL_PREFIX}")
endif()

if (BUILD_PYTHON)
    pybind11_add_module(pySDFSampler SHARED python/python.cpp)

    target_link_libraries(pySDFSampler PUBLIC Sampler)

    set_target_properties(pySDFSampler PROPERTIES OUTPUT_NAME SDFSampler)

    add_dependencies(pySDFSampler Sampler)
    
    install(TARGETS 
        pySDFSampler
        LIBRARY DESTINATION ./
        RUNTIME DESTINATION ./)

    install(FILES 
            "${CMAKE_SOURCE_DIR}/OverfitShapes/NeuralImplicit.py"
            "${CMAKE_SOURCE_DIR}/OverfitShapes/FileUtilities.py"
            "${CMAKE_SOURCE_DIR}/OverfitShapes/Overfit_Neural_Networks_as_a_Compact_Shape_Representation.ipynb"
        DESTINATION "${CMAKE_INSTALL_PREFIX}")

    if (BUILD_EXAMPLE)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/example/example.py"
                DESTINATION "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

install(TARGETS 
    Sampler
    LIBRARY DESTINATION ./
    RUNTIME DESTINATION ./)

get_target_property(TBB_LOCATION TBB::tbb IMPORTED_LOCATION_RELEASE)
install(FILES "${TBB_LOCATION}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}")
