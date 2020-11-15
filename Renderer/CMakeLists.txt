project(Renderer LANGUAGES CXX CUDA)

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

    add_dependencies(RendererExample Renderer)

    install(TARGETS
        RendererExample
        LIBRARY DESTINATION ./
        RUNTIME DESTINATION ./)

    install(FILES
        example/weights.bin
        example/biases.bin
        DESTINATION ./)
endif()