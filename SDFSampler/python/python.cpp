#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "Octree/Octree.h"
#include "IO/MeshLoader.h"
#include "SDFSampler/PointSampler.h"

using namespace SDFSampler;
namespace py = pybind11;
PYBIND11_MODULE(SDFSampler, m) {
    py::class_<Octree>(m, "Octree")
        .def(py::init<const Eigen::Ref<const Eigen::MatrixXf>&,
            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>&>())
        .def("closestFace", &Octree::closestFace)
        .def("closestPoint", &Octree::closestPoint);

    py::class_<MeshLoader>(m, "MeshLoader")
        .def(py::init<>())
        .def("read", &MeshLoader::read);

    py::class_<PointSampler>(m, "PointSampler")
        .def(py::init<const Eigen::Ref<const Eigen::MatrixXf>,
            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>,
            int>(), py::arg("vertices"), py::arg("faces"), py::arg("seed") = -1)
        .def("sample", &PointSampler::sample, py::arg("numPoints") = 1, py::arg("sampleSetScale") = 10)
        .def("setImportanceThreshold", &PointSampler::setImportanceThreshold)
        .def("boundingSphere", &PointSampler::boundingVolume)
        .def_readwrite("importanceThreshold", &PointSampler::importance_threshold_)
        .def_readwrite("beta", &PointSampler::beta_);
}