#include <SDFSampler/PointSampler.h>

#include "UniformSampleNBall.h"

#include "FixedPriorityQueue.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include <iostream>

using namespace SDFSampler;

PointSampler::PointSampler(const Eigen::Ref<const Eigen::MatrixXf> vertices,
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces,
    int seed)
    : vertices_(vertices),
    faces_(faces),
    importance_threshold_(1.0e-4),
    beta_(30),
    tree_(vertices, faces)
{
    assert(vertices.cols() == 3);
    assert(vertices.rows() > 0);
    assert(faces.rows() > 0);

    importance_func_ = [&](const Eigen::Vector3f& point, float dist) -> float
    {
        return exp(-beta_ * abs(dist));
    };

    if (seed < 0)
    {
        std::random_device rd;
        seed_ = rd();
    }
    else
    {
        seed_ = seed;
    }

    mean_ = vertices.colwise().mean();
    bounding_radius_ = (vertices.rowwise() - mean_.transpose()).rowwise().norm().maxCoeff();

    std::vector<HDK_Sample::UT_Vector3T<float> > U(vertices.rows());
    for (size_t i = 0; i < vertices.rows(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            U[i][j] = vertices(i, j);
        }
    }

    solid_angle_.init(faces.rows(), faces.data(), vertices.rows(), &U[0]);
}

std::pair<Eigen::Vector3f, float>
PointSampler::boundingSphere()
{
    return std::make_pair(mean_, bounding_radius_);
}

std::pair<Eigen::MatrixXf, Eigen::VectorXf>
PointSampler::sample(const size_t numPoints, const float sampleSetScale)
{
    assert(sampleSetScale >= 1);
    std::pair<Eigen::MatrixXf, Eigen::VectorXf> result;
    float radius = bounding_radius_ + importance_threshold_;

    auto greater = [&](const std::pair<Eigen::Vector3f, float>& pt1, const std::pair<Eigen::Vector3f, float>&pt2) {
        return importance_func_(pt1.first, pt1.second) > importance_func_(pt2.first, pt2.second);
    };

    FixedMinPriorityQueue<std::pair<Eigen::Vector3f, float>, decltype(greater)> queue(numPoints, greater);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numPoints*(double)sampleSetScale), [&](tbb::blocked_range<size_t> r)
    {
        for (size_t i = r.begin(); i < r.end(); ++i)
        {
            float importance;
            HDK_Sample::UT_Vector3T<float> point;
            nrg::UniformSampleNBall<3>(radius, point, seed_);
            point[0] += mean_[0];
            point[1] += mean_[1];
            point[2] += mean_[2];



            float winding_num = solid_angle_.computeSolidAngle(point) / (4.0 * M_PI);

            Eigen::Vector3f pt = Eigen::Vector3f(point[0], point[1], point[2]);

            float dist = tree_.closestFace(pt).second;
            if (winding_num > 0)
            {
                dist *= -1;
            }

            queue.push(std::make_pair(pt, dist));
        }
    });

    result.first.resize(numPoints, 3);
    result.second.resize(numPoints);
    const auto& data = queue.data();

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numPoints), [&](tbb::blocked_range<size_t> r)
    {
        for (size_t i = r.begin(); i < r.end(); ++i)
        {
            result.first.row(i) = data[i].first;
            result.second(i) = data[i].second;
        }
    });

    return result;
}
