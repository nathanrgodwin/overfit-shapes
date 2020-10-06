#pragma once
#define _USE_MATH_DEFINES
#include <cmath>

#include <WindingNumber/UT_SolidAngle.h>

#include "UniformSampleNBall.h"

#include <Eigen/Core>

#include <Octree/Octree.h>

#include "ExportSemantics.h"

namespace SDFSampler
{
class EXPORT PointSampler
{
public:
    PointSampler(const Eigen::Ref<const Eigen::MatrixXf> vertices,
        const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces,
        int seed = -1);

    std::pair<Eigen::Vector3f, float>
    boundingVolume();

    std::pair<Eigen::MatrixXf, Eigen::VectorXf>
    sample(const size_t numPoints = 1, const float sampleSetScale = 10);

    inline const unsigned int
    seed() const
    {
        return seed_;
    }

    inline void
    setSeed(unsigned int seed)
    {
        seed_ = seed;
    }

    inline std::function<float(const Eigen::Vector3f&, float)>&
    importanceFunction()
    {
        return importance_func_;
    }

    inline void
    setImportanceThreshold(float threshold)
    {
        importance_threshold_ = threshold;
    }

    inline float&
    setBeta()
    {
        return beta_;
    }

    float beta_;
    float importance_threshold_;

private:
    Octree tree_;
    HDK_Sample::UT_SolidAngle<float, float> solid_angle_;
    std::function<float(const Eigen::Vector3f&, float)> importance_func_;
    Eigen::Vector3f mean_;
    unsigned int seed_;
    float bounding_radius_;
    const Eigen::Ref<const Eigen::MatrixXf> vertices_;
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces_;
};
}