#include "SDFSampler/PointSampler.h"
#include "IO/MeshLoader.h"
#include "FixedPriorityQueue.h"

#include <iostream>

int
main(int argc, char** argv)
{
    Eigen::MatrixXf vertices;
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> faces;
    std::tie(vertices, faces) = MeshLoader::read("cube.obj");//"bunny.stl");
    SDFSampler::PointSampler sampler(vertices, faces);
    std::cout << sampler.sample(10, 20).first << std::endl;
}