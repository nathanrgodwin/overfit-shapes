#pragma once
#define _USE_MATH_DEFINES
#include <cmath>

#include <WindingNumber/UT_SolidAngle.h>

#include "UniformSampleNBall.h"

#include <Eigen/Core>

#include <AABB_tree/AABB_tree.h>

#include "ExportSemantics.h"
#include "MeshReference.h"

namespace SDFSampler
{
class EXPORT PointSampler
{
public:
    PointSampler(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
        const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces,
        int seed = -1);

    std::pair<Eigen::MatrixXf, Eigen::VectorXf>
    sample(const size_t numPoints = 1, const float sampleSetScale = 10);

    inline unsigned int&
    seed()
    {
        return seed_;
    }

    inline std::function<float(const Eigen::Vector3f&, float)>&
    importanceFunction()
    {
        return importance_func_;
    }

    inline float&
    beta()
    {
        return beta_;
    }

    float beta_;
    unsigned int seed_;

private:
    std::unique_ptr<AABB_tree<float>> tree_;
    HDK_Sample::UT_SolidAngle<float, float> solid_angle_;
    std::function<float(const Eigen::Vector3f&, float)> importance_func_;
    std::shared_ptr<MeshReference> mesh_;
};
}