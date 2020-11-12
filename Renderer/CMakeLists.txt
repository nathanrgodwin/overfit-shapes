project(Renderer LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(NvidiaCutlass REQUIRED)

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif()

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()

add_library(Renderer SHARED
    kernel.cu)

target_compile_features(Renderer PUBLIC cxx_std_11)

target_include_directories(Renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${UTILITY_DIR})

target_link_libraries(Renderer PRIVATE
    nvidia::cutlass::cutlass)

install(TARGETS
    Renderer
    LIBRARY DESTINATION ./
    RUNTIME DESTINATION ./)

if (BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED)

    add_executable(RendererExample example/main.cpp)

    target_link_libraries(RendererExample PUBLIC
        Renderer
        ${OpenCV_LIBRARIES})

    target_include_directories(RendererExample PUBLIC
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR})

    install(TARGETS
        RendererExample
        LIBRARY DESTINATION ./
        RUNTIME DESTINATION ./)
endif()