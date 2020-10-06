#include <Octree/Octree.h>

#include <vector>
#include <numeric>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

/**
* This is a pretty naive approach
**/

Octree::Octree(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces)
    : vertices_(vertices), faces_(faces)
{
    vert_face_assoc = std::unique_ptr<std::unordered_map<size_t, std::set<size_t>>>(new std::unordered_map<size_t, std::set<size_t>>());
    for (size_t i = 0; i < faces.rows(); ++i)
    {
        (*vert_face_assoc)[faces.row(i)[0]].insert(i);
        (*vert_face_assoc)[faces.row(i)[1]].insert(i);
        (*vert_face_assoc)[faces.row(i)[2]].insert(i);
    }

    std::vector<size_t> vertex_idx(vertices.rows());
    std::iota(vertex_idx.begin(), vertex_idx.end(), 0);
    root.init(vertices, faces, vertex_idx);
}

Octree::OctreeNode::OctreeNode(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces,
    const std::vector<size_t>& vertex_idx)
{
    init(vertices, faces, vertex_idx);
}

/**
* 0: x+, y+, z+
* 1: x-, y+, z+
* 2: x+, y-, z+
* 3: x-, y-, z+
* ...
**/
bool
Octree::OctreeNode::checkRegion(const Eigen::Vector3f& pt, unsigned char region) const
{
    return ((!(region & 0x1) && pt[0] >= center[0]) || ((region & 0x1) && pt[0] < center[0])) &&
        ((!(region & 0x2) && pt[1] >= center[1]) || ((region & 0x2) && pt[1] < center[1])) &&
        ((!(region & 0x4) && pt[2] >= center[2]) || ((region & 0x4) && pt[2] < center[2]));

}

void
Octree::OctreeNode::init(const Eigen::Ref<const Eigen::MatrixXf>& vertices,
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces,
    const std::vector<size_t>& vertex_idx)
{
    if (vertex_idx.size() == 1)
    {
        idx = vertex_idx[0];
        center = vertices.row(idx);
        return;
    }
    for (const auto& idx : vertex_idx)
    {
        center += vertices.row(idx);
    }
    center /= (float)vertex_idx.size();

    //Parallelize Octree construction
    tbb::parallel_for(tbb::blocked_range<unsigned char>(0, 8), [&](tbb::blocked_range<unsigned char> r)
    {
        for (unsigned char i = r.begin(); i < r.end(); ++i)
        {
            std::vector<size_t> indices;
            for (const auto& idx : vertex_idx)
            {
                if (checkRegion(vertices.row(idx), i))
                {
                    indices.push_back(idx);
                }
            }
            if (!indices.empty())
            {
                children[i] = new OctreeNode(vertices, faces, indices);
                has_children = true;
            }
        }
    });
}


std::pair<size_t, float>
Octree::closestFace(const Eigen::Vector3f& pt) const
{
    size_t pt_idx = root.closestPoint(pt).first;
    const std::set<size_t>& faces_idx = vert_face_assoc->at(pt_idx);
    size_t min_face_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (const auto& idx : faces_idx)
    {
        float dist = pointToTriangleDist(pt, idx);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_face_idx = idx;
        }
    }
    return std::make_pair(min_face_idx, min_dist);
}

// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2020
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 4.0.2019.08.13
float
Octree::pointToTriangleDist(const Eigen::Vector3f& pt, const size_t triangle_idx) const
{
    Eigen::Vector3f v0 = vertices_.row(faces_.row(triangle_idx)[0]);
    Eigen::Vector3f v1 = vertices_.row(faces_.row(triangle_idx)[1]);
    Eigen::Vector3f v2 = vertices_.row(faces_.row(triangle_idx)[2]);
    Eigen::Vector3f diff = pt - v0;
    Eigen::Vector3f edge0 = v1 - v0;
    Eigen::Vector3f edge1 = v2 - v0;
    float a00 = edge0.dot(edge0);
    float a01 = edge0.dot(edge1);
    float a11 = edge1.dot(edge1);
    float b0 = -diff.dot(edge0);
    float b1 = -diff.dot(edge1);
    float det = a00 * a11 - a01 * a01;
    float t0 = a01 * b1 - a11 * b0;
    float t1 = a01 * b0 - a00 * b1;

    if (t0 + t1 <= det)
    {
        if (t0 < 0)
        {
            if (t1 < 0)  // region 4
            {
                if (b0 < 0)
                {
                    t1 = 0;
                    if (-b0 >= a00)  // V1
                    {
                        t0 = 1;
                    }
                    else  // E01
                    {
                        t0 = -b0 / a00;
                    }
                }
                else
                {
                    t0 = 0;
                    if (b1 >= 0)  // V0
                    {
                        t1 = 0;
                    }
                    else if (-b1 >= a11)  // V2
                    {
                        t1 = 1;
                    }
                    else  // E20
                    {
                        t1 = -b1 / a11;
                    }
                }
            }
            else  // region 3
            {
                t0 = 0;
                if (b1 >= 0)  // V0
                {
                    t1 = 0;
                }
                else if (-b1 >= a11)  // V2
                {
                    t1 = 1;
                }
                else  // E20
                {
                    t1 = -b1 / a11;
                }
            }
        }
        else if (t1 < 0)  // region 5
        {
            t1 = 0;
            if (b0 >= 0)  // V0
            {
                t0 = 0;
            }
            else if (-b0 >= a00)  // V1
            {
                t0 = 1;
            }
            else  // E01
            {
                t0 = -b0 / a00;
            }
        }
        else  // region 0, interior
        {
            float invDet = 1 / det;
            t0 *= invDet;
            t1 *= invDet;
        }
    }
    else
    {
        float tmp0, tmp1, numer, denom;

        if (t0 < 0)  // region 2
        {
            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - (float)2 * a01 + a11;
                if (numer >= denom)  // V1
                {
                    t0 = 1;
                    t1 = 0;
                }
                else  // E12
                {
                    t0 = numer / denom;
                    t1 = 1 - t0;
                }
            }
            else
            {
                t0 = 0;
                if (tmp1 <= 0)  // V2
                {
                    t1 = 1;
                }
                else if (b1 >= 0)  // V0
                {
                    t1 = 0;
                }
                else  // E20
                {
                    t1 = -b1 / a11;
                }
            }
        }
        else if (t1 < 0)  // region 6
        {
            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - (float)2 * a01 + a11;
                if (numer >= denom)  // V2
                {
                    t1 = 1;
                    t0 = 0;
                }
                else  // E12
                {
                    t1 = numer / denom;
                    t0 = 1 - t1;
                }
            }
            else
            {
                t1 = 0;
                if (tmp1 <= 0)  // V1
                {
                    t0 = 1;
                }
                else if (b0 >= 0)  // V0
                {
                    t0 = 0;
                }
                else  // E01
                {
                    t0 = -b0 / a00;
                }
            }
        }
        else  // region 1
        {
            numer = a11 + b1 - a01 - b0;
            if (numer <= 0)  // V2
            {
                t0 = 0;
                t1 = 1;
            }
            else
            {
                denom = a00 - (float)2 * a01 + a11;
                if (numer >= denom)  // V1
                {
                    t0 = 1;
                    t1 = 0;
                }
                else  // 12
                {
                    t0 = numer / denom;
                    t1 = 1 - t0;
                }
            }
        }
    }

    Eigen::Vector3f closest = v0 + t0 * edge0 + t1 * edge1;
    return (pt - closest).norm();

}

std::pair<size_t,float>
Octree::closestPoint(const Eigen::Vector3f& pt) const
{
    return root.closestPoint(pt);
}

std::pair<size_t, float>
Octree::OctreeNode::closestPoint(const Eigen::Vector3f& pt) const
{
    if (!has_children) return std::make_pair(idx, (pt - center).norm());
    int close_idx = 0;
    float close_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < 8; ++i)
    {
        if (children[i] != nullptr)
        {
            float dist = (pt - children[i]->center).norm();
            if (dist < close_dist)
            {
                close_idx = i;
                close_dist = dist;
            }
        }
    }
    return children[close_idx]->closestPoint(pt);
}

Octree::OctreeNode::~OctreeNode()
{
    if (!has_children) return;
    for (int i = 0; i < 8; ++i)
    {
        if (children[i] != nullptr) delete children[i];
    }
}
