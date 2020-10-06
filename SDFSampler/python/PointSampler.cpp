#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "SDFSampler/PointSampler.h"

using namespace SDFSampler;

namespace py = pybind11;
PYBIND11_MODULE(SDFSampler, m) {
    py::class_<PointSampler>(m, "PointSampler")
        .def(py::init<const Eigen::Ref<const Eigen::MatrixXf>,
            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>,
            int>(), py::arg("vertices"), py::arg("faces"), py::arg("seed") = -1)
        .def("sample", &PointSampler::sample, py::arg("numPoints") = 1)
        .def("setImportanceThreshold", &PointSampler::setImportanceThreshold)
        .def_readwrite("importanceThreshold", &PointSampler::importance_threshold_)
        .def_readwrite("beta", &PointSampler::beta_);
}