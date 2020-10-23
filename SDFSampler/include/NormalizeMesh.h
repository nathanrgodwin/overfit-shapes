#pragma once

#include <Eigen/Core>
#include "ExportSemantics.h"

static void
normalizeMeshToUnitSphere(Eigen::Ref<Eigen::MatrixXf> vertices,
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces)
{
    Eigen::Matrix<float, 1, 3> mean = vertices.colwise().mean();
    vertices.rowwise() -= mean;

    Eigen::MatrixXf::Index idx;
    float scale_factor = vertices.rowwise().norm().maxCoeff(&idx);

    Eigen::Matrix<float, 1, 3> scale = vertices.row(idx);

    vertices.array().rowwise() /= (scale_factor * scale).array();
}