#pragma once

#include <Eigen/Core>
#include "ExportSemantics.h"

static std::pair<Eigen::Vector3f, float>
normalizeMeshToUnitSphere(Eigen::Ref<Eigen::MatrixXf> vertices,
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces)
{
    assert(vertices.rows() > 0);
    assert(faces.rows() > 0);
    Eigen::Matrix<float, 1, 3> mean = vertices.colwise().mean();
    vertices.rowwise() -= mean;

    float scale_factor = vertices.rowwise().norm().maxCoeff();

    vertices /= scale_factor;
    return std::make_pair(mean, scale_factor);
}