#include "SDFSampler/PointSampler.h"
#include "IO/MeshLoader.h"
#include "FixedPriorityQueue.h"

#include <iostream>

int
main(int argc, char** argv)
{
    /*Eigen::MatrixXf vertices(8, 3);
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> faces(11,3);

    vertices << 0.0, 0.0, 0.0,
        0.0,  0.0,  1.0,
        0.0,  1.0,  0.0,
        0.0,  1.0,  1.0,
        1.0,  0.0,  0.0,
        1.0,  0.0,  1.0,
        1.0,  1.0,  0.0,
        1.0,  1.0,  1.0;

    faces << 0, 6, 4,
        0, 2, 6,
        0, 3, 2,
        0, 1, 3,
        2, 7, 6,
        2, 3, 7,
        4, 6, 7,
        4, 7, 5,
        0, 4, 5,
        0, 5, 1,
        1, 5, 7;/* ,
        1, 7, 3;*/

        /*SDFSampler::PointSampler sampler(vertices, faces);
        auto pt_data = sampler.sample(1e6);
        std::cout << pt_data.first << "\n\n" << pt_data.second <<  std::endl;*/

    Eigen::MatrixXf vertices;
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> faces;
    std::tie(vertices, faces) = MeshLoader::read("bunny.stl");
    SDFSampler::PointSampler sampler(vertices, faces);
    std::cout << sampler.sample(1e6, 20).first << std::endl;


}