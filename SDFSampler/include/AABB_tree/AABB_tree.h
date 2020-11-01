#pragma once

#include <numeric>
#include <unordered_map>
#include <set>
#include <Eigen/Dense>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>

#include "MeshReference.h"

//Wrapper for CGAL AABB_tree with Eigen data
template <typename T>
class AABB_tree
{
private:
    typedef CGAL::Simple_cartesian<T> K;

    class MeshIterator
    {
        size_t idx_;
    public:
        std::shared_ptr<MeshReference> mesh_;

        using iterator_category = std::forward_iterator_tag;
        using value_type = size_t;
        using difference_type = size_t;
        using pointer = size_t;
        using reference = size_t;

        MeshIterator() {}
        MeshIterator(std::shared_ptr<MeshReference>& mesh, size_t idx) : mesh_(mesh), idx_(idx) {}

        pointer
        data() { return idx_; }

        reference
        operator*() { return idx_; }

        const value_type
        operator*() const { return idx_; }

        bool
        operator!=(const MeshIterator& other)
        {
            return idx_ != (*other);
        }

        MeshIterator&
        operator++()
        {
            idx_ += 1;
            return *this;
        }

        MeshIterator
        operator++(int)
        {
            return MeshIterator(mesh_, idx_ + 1);
        }

        MeshIterator&
        operator=(const MeshIterator& other)
        {
            mesh_ = other.mesh_;
            idx_ = other.idx_;
            return *this;
        }
    };

    struct TrianglePrimative
    {
    public:
        typedef size_t Id;
        typedef typename AABB_tree<T>::K::Point_3 Point;
        typedef typename AABB_tree<T>::K::Triangle_3 Datum;

    private:
        Id idx_;
        MeshIterator iterator_;

    public:
        TrianglePrimative(MeshIterator iterator) : iterator_(iterator), idx_(*iterator) {}

        const Id&
        id() const { return idx_; }

        Point
        convert(const Id idx) const
        {
            const auto& row = iterator_.mesh_->vertices.row(idx);
            return Point(row[0], row[1], row[2]);
        }

        Datum
        datum() const
        {
            const auto& row = iterator_.mesh_->faces.row(idx_);
            return Datum(convert(row[0]), convert(row[1]), convert(row[2]));
        }

        Point
        reference_point() const
        {
            const auto& row = iterator_.mesh_->faces.row(idx_);
            return convert(row[0]);
        }

        TrianglePrimative&
        operator=(const TrianglePrimative& other)
        {
            idx_ = other.idx_;
            iterator_ = other.iterator_;
            return *this;
        }
    };

    typedef CGAL::AABB_traits<K, TrianglePrimative> TreeTraits;
    typedef CGAL::AABB_tree<TreeTraits> Tree;

    std::unique_ptr<Tree> tree_;

    T
    dist(const Eigen::Matrix<T, 3, 1>& pt1, const Eigen::Matrix<T, 3, 1>& pt2) const
    {
        return (pt1 - pt2).norm();
    }

    std::shared_ptr<MeshReference> mesh_;

public:

    AABB_tree(std::shared_ptr<MeshReference>& mesh)
        : mesh_(mesh)
    {
        auto begin = MeshIterator(mesh_, 0);
        auto end = MeshIterator(mesh_, mesh_->faces.rows());
        tree_ = std::unique_ptr<Tree>(new Tree(begin, end));
        tree_->accelerate_distance_queries();
    }

	AABB_tree(const Eigen::Ref<const Eigen::Matrix<T, -1, -1>>& vertices,
		const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& faces)
		: AABB_tree(std::make_shared<MeshReference>(vertices, faces))
	{}

    std::pair<Eigen::Matrix<T, 3, 1>, T>
    closestPoint(const Eigen::Matrix<T, 3, 1>& pt) const
    {
        K::Point_3 closest = tree_->closest_point(K::Point_3(pt[0], pt[1], pt[2]));
        Eigen::Matrix<T, 3, 1> close_eigen(closest.x(), closest.y(), closest.z());
        return std::make_pair(close_eigen, dist(close_eigen, pt));
    }

	std::pair<typename AABB_tree<T>::TrianglePrimative::Id, T>
	closestFace(const Eigen::Matrix<T, 3, 1>& pt) const
	{
        auto closest = tree_->closest_point_and_primitive(K::Point_3(pt[0], pt[1], pt[2]));
        Eigen::Matrix<T, 3, 1> close_eigen(closest.first.x(), closest.first.y(), closest.first.z());
        return std::make_pair(closest.second, dist(close_eigen, pt));
	}

};