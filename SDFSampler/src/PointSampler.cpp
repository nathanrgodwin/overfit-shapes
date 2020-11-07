#include <SDFSampler/PointSampler.h>

#include "UniformSampleNBall.h"

#include "FixedPriorityQueue.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include <iostream>

using namespace SDFSampler;

PointSampler::PointSampler(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces,
    int seed)
    : mesh_(std::make_shared<MeshReference>(vertices, faces)),
    beta_(30.0f)
{
    assert(vertices.cols() == 3);
    assert(vertices.rows() > 0);
    assert(faces.rows() > 0);

    tree_ = std::unique_ptr<AABB_tree<float>>(new AABB_tree<float>(mesh_));

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



    std::vector<HDK_Sample::UT_Vector3T<float> > U(vertices.rows());
    for (int i = 0; i < vertices.rows(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            U[i][j] = vertices(i, j);
        }
    }

    solid_angle_.init(faces.rows(), faces.data(), vertices.rows(), &U[0]);
}

std::pair<Eigen::MatrixXf, Eigen::VectorXf>
PointSampler::sample(const size_t numPoints, const float sampleSetScale)
{
    assert(sampleSetScale >= 1);
    std::pair<Eigen::MatrixXf, Eigen::VectorXf> result;

    auto greater = [&](const std::pair<Eigen::Vector3f, float>& pt1, const std::pair<Eigen::Vector3f, float>&pt2) {
        return importance_func_(pt1.first, pt1.second) > importance_func_(pt2.first, pt2.second);
    };

    FixedMinPriorityQueue<std::pair<Eigen::Vector3f, float>, decltype(greater)> queue(numPoints, greater);
    float zero[3]{ 0,0,0 };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(numPoints*(double)sampleSetScale)), [&](tbb::blocked_range<size_t> r)
    {
        for (size_t i = r.begin(); i < r.end(); ++i)
        {
            HDK_Sample::UT_Vector3T<float> point;
            nrg::UniformSampleNBall<3>(1.0f, zero, seed_, point, tbb::this_task_arena::current_thread_index());

            float winding_num = solid_angle_.computeSolidAngle(point) / (4.0f * (float)M_PI);

            Eigen::Vector3f pt = Eigen::Vector3f(point[0], point[1], point[2]);

            float dist = tree_->closestPoint(pt).second;
            if (winding_num > 1.2e-3) //Experimentally determined for cube.obj, unknown for other mesh
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
