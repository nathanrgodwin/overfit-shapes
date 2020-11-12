#pragma once

#include "kernel.cuh"
#include <Eigen/Dense>
#include "ExportSemantics.h"

class EXPORT SDFRenderer : public Renderer
{
public:
	SDFRenderer() : Renderer() {}

    SDFRenderer(unsigned int num_layers, unsigned int layer_size,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& weights,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& biases) : Renderer()
    {
        setModel(num_layers, layer_size, weights, biases);
    }

    inline void
    setModel(unsigned int num_layers, unsigned int layer_size,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& weights,
        const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>& biases)
    {
        params_.num_layers = num_layers;
        params_.layer_size = layer_size;
        size_t num_weights = 4 * layer_size + num_layers * num_layers * layer_size;
        size_t num_biases = layer_size * num_layers + 1;

        if (params_.weights) gpuDelete(params_.weights);
        if (params_.biases) gpuDelete(params_.biases);

        copyDataToGPU(&params_.weights, weights.data(), num_weights);
        copyDataToGPU(&params_.biases, biases.data(), num_biases);
    }


    inline Eigen::Matrix<unsigned char, -1, -1, Eigen::RowMajor>
    render()
    {
        Eigen::Matrix<unsigned char, -1, -1, Eigen::RowMajor> image(params_.height, 3*params_.width);
        params_.image = image.data();
        Renderer::render();
        return image;
    }

};