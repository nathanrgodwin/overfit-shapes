#pragma once

#include "kernel.cuh"
#include <Eigen/Dense>
#include "ExportSemantics.h"

#include <iostream>

struct Image : Eigen::Matrix<unsigned char, -1, -1, Eigen::RowMajor>
{
public:
    Image(unsigned int width, unsigned int height)
       : Eigen::Matrix<unsigned char, -1, -1, Eigen::RowMajor>(height, width*3) {}

};

class EXPORT SDFRenderer : public Renderer
{
public:
	SDFRenderer() : Renderer() {}

    SDFRenderer(unsigned int H, unsigned int N,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& weights,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& biases) : Renderer()
    {
        setModel(H, N, weights, biases);
    }

    inline void
    setModel(unsigned int H, unsigned int N,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& weights,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& biases)
    {

        /**
        * Maximum supported by renderer currently. Reduced performance after 128.
        *
        * This is because I allocate buffers on shared memory, and there is 48kb
        * shared memory per block. Launching a block of size < 32 wastes threads in the warp,
        * launching 1 block per SM wastes SM, so target block size of 32, shared memory size
        * of 32kb per block (allowing min. 2 blocks per SM).
        *
        * Reduced performance after:
        * 32*1024/(2*N*sizeof(float)) = 32 => N = 128
        *
        * Out of memory after:
        * 48*1024/(2*N*sizeof(float)) = 32 => N = 192
        **/
        if (N > 128 && N <= 192) std::cout << "Renderer performance is reduced after layer size of 128" << std::endl;
        if (N > 192) throw std::runtime_error("This renderer supports a maximum layer size of 192");

        params_.H = H;
        params_.N = N;
        size_t num_weights = 4 * N + N * N * (H - 1);
        size_t num_biases = N * H + 1;

        if (params_.weights) gpuDelete(params_.weights);
        if (params_.biases) gpuDelete(params_.biases);

        copyDataToGPU(&params_.weights, weights.data(), num_weights);
        copyDataToGPU(&params_.biases, biases.data(), num_biases);
    }


    inline Image
    render()
    {
        if (params_.weights == nullptr || params_.biases == nullptr) throw std::runtime_error("Renderer model is not initialized");
        Image img(params_.width, params_.height);
        params_.image = img.data();
        Renderer::render();
        return img;
    }

};