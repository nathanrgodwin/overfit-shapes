#pragma once

#include <Eigen/Core>
#include "ExportSemantics.h"

static std::pair<Eigen::Vector3f, float>
normalizeMeshToSphere(Eigen::Ref<Eigen::MatrixXf> vertices,
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces,
    float radius = 1)
{
    assert(vertices.rows() > 0);
    assert(faces.rows() > 0);
    Eigen::Matrix<float, 1, 3> mean = vertices.colwise().mean();
    vertices.rowwise() -= mean;

    float scale_factor = vertices.rowwise().norm().maxCoeff()
                            * (1.0f / radius);

    vertices /= scale_factor;
    return std::make_pair(mean, scale_factor);
}