#pragma once

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <unordered_map>
#include <memory>

#include "ExportSemantics.h"

class EXPORT Octree
{
public:
    Octree(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
        const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces);

    std::pair<size_t, float>
    closestFace(const Eigen::Vector3f& pt) const;

    std::pair<size_t, float>
    closestPoint(const Eigen::Vector3f& pt) const;

private:
    std::unique_ptr<std::unordered_map<size_t, std::set<size_t>>> vert_face_assoc;
    const Eigen::Ref<const Eigen::MatrixXf> vertices_;
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces_;

    float
    pointToTriangleDist(const Eigen::Vector3f& pt, const size_t triangle_idx) const;

    class EXPORT OctreeNode
    {
    public:
        OctreeNode() {}

        OctreeNode(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces,
            const std::vector<size_t>& vertex_idx);

        ~OctreeNode();

        std::pair<size_t, float>
        closestPoint(const Eigen::Vector3f& pt) const;


        size_t idx = 0;
        OctreeNode* children[8] = { nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr };
        bool has_children = false;
        Eigen::Vector3f center = Eigen::Vector3f(0, 0, 0);

        void
        init(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces,
            const std::vector<size_t>& vertex_idx);

        bool
        checkRegion(const Eigen::Vector3f& pt, unsigned char region) const;
    };

    OctreeNode root;
};

