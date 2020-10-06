#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "IO/MeshLoader.h"

namespace py = pybind11;
PYBIND11_MODULE(SDFSampler, m) {
    py::class_<MeshLoader>(m, "MeshLoader")
        .def(py::init<>())
        .def("read", &MeshLoader::read)
        .def_readwrite("vertices", &MeshLoader::vertices_)
        .def_readwrite("faces", &MeshLoader::faces_);
}