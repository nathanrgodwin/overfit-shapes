#include "SDFSampler/PointSampler.h"
#include "IO/MeshLoader.h"
#include "NormalizeMesh.h"

#include <iostream>

int
main(int argc, char** argv)
{
    Eigen::MatrixXf vertices;
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> faces;
    std::tie(vertices, faces) = MeshLoader::read("cube.obj");
    std::cout << vertices.rows() << ", " << faces.rows() << std::endl;
    normalizeMeshToUnitSphere(vertices, faces);
    SDFSampler::PointSampler sampler(vertices, faces);
    std::cout << sampler.sample(10, 20).first << std::endl;
}