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
        params_.H = H;
        params_.N = N;
        size_t num_weights = 4 * N + N * N * (H - 1);
        size_t num_biases = N * H + 1;

        //std::cout << H << ", " << N << ", " << weights.rows() << ", " << biases.rows() << std::endl;

        if (params_.weights) gpuDelete(params_.weights);
        if (params_.biases) gpuDelete(params_.biases);

        copyDataToGPU(&params_.weights, weights.data(), num_weights);
        copyDataToGPU(&params_.biases, biases.data(), num_biases);
    }


    inline Image
    render()
    {
        Image img(params_.width, params_.height);
        params_.image = img.data();
        Renderer::render();
        return img;
    }

};