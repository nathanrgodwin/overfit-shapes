#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "AABB_tree/AABB_tree.h"
#include "IO/MeshLoader.h"
#include "SDFSampler/PointSampler.h"
#include "NormalizeMesh.h"

#ifdef BUILD_RENDERER
#include "Renderer.h"
#include "ImageCaster.h"
#endif

using namespace SDFSampler;
namespace py = pybind11;
PYBIND11_MODULE(OverfitShapes, m) {


    py::class_<AABB_tree<float>>(m, "AABB_tree")
        .def(py::init<const Eigen::Ref<const Eigen::MatrixXf>&,
            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>&>())
        .def("closestFace", &AABB_tree<float>::closestFace)
        .def("closestPoint", &AABB_tree<float>::closestPoint);

    py::class_<MeshLoader>(m, "MeshLoader")
        .def(py::init<>())
        .def("read", &MeshLoader::read);

    py::class_<PointSampler>(m, "PointSampler")
        .def(py::init<const Eigen::Ref<const Eigen::MatrixXf>,
            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>,
            int>(), py::arg("vertices"), py::arg("faces"), py::arg("seed") = -1)
        .def("sample", &PointSampler::sample, py::arg("numPoints") = 1, py::arg("sampleSetScale") = 10)
        .def_readwrite("beta", &PointSampler::beta_)
        .def_readwrite("seed", &PointSampler::seed_);

    m.def("normalizeMeshToSphere", &normalizeMeshToSphere,
        py::arg("vertices"), py::arg("faces"), py::arg("radius") = 1,
        "Normalizes a mesh to the unit sphere at origin");

#ifdef BUILD_RENDERER
    py::class_<Light>(m, "Light")
        .def(py::init<>())
        .def_property("color", &Light::getColor, &Light::setColor)
        .def_property("position", &Light::getPosition, &Light::setPosition)
        .def_property("strength", &Light::getStrength, &Light::setStrength)
        .def_property_readonly("ambient_strength", &Light::ambientStrength)
        .def_property_readonly("diffuse_strength", &Light::diffuseStrength)
        .def_property("specular_power", &Light::getSpecularPower, &Light::setSpecularPower)
        .def_property_readonly("specular_strength", &Light::specularStrength);

    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def_property("position", &Camera::getPosition, &Camera::setPosition)
        .def_property("direction", &Camera::getDirection, &Camera::setDirection)
        .def_property_readonly("up", &Camera::getUp)
        .def_property("side", &Camera::getSide, &Camera::setSide)
        .def_property("fov", &Camera::getFOV, &Camera::setFOV)
        .def_property_readonly("fov_scale", &Camera::fovScale)
        .def_property("up", &Camera::getMaxDist, &Camera::setMaxDist);

    py::class_<SDFRenderer>(m, "Renderer")
        .def(py::init<>())
        .def(py::init<unsigned int, unsigned int,
            const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>&,
            const Eigen::Ref<const Eigen::Matrix<float, -1, 1>>&>())
        .def("setModel", &SDFRenderer::setModel)
        .def("render", &SDFRenderer::render, py::return_value_policy::move)
        .def_property("resolution", &SDFRenderer::getResolution, &SDFRenderer::setResolution)
        .def_property("min_dist", &SDFRenderer::getMinDist, &SDFRenderer::setMinDist)
        .def_property("eps", &SDFRenderer::getEps, &SDFRenderer::setEps)
        .def_property("light", &SDFRenderer::getLight, &SDFRenderer::setLight)
        .def_property("camera", &SDFRenderer::getCamera, &SDFRenderer::setCamera)
        .def_property("background_color", &SDFRenderer::getBackgroundColor, &SDFRenderer::setBackgroundColor);
#endif
}
