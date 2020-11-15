#include "Renderer.h"

#include <fstream>

#include <opencv2/opencv.hpp>

inline float
leakyReLU(float val, float slope)
{
	return fmaxf(0, val) + slope * fminf(0, val);
}

int main()
{
    std::ifstream weight_file("weights.bin", std::ios_base::binary);
    std::ifstream bias_file("biases.bin", std::ios_base::binary);

    if (!weight_file.is_open())
    {
        std::cerr << "Unable to open weight file" << std::endl;
        return 1;
    }
    if (!bias_file.is_open())
    {
        std::cerr << "Unable to open bias file" << std::endl;
        return 1;
    }

    unsigned int N = 32;
    unsigned int H = 8;
    size_t num_weights, num_biases;
    num_weights = 3 * N + N * N * (H - 1) + N;
    num_biases = N * H + 1;
    Eigen::Matrix<float, Eigen::Dynamic, 1> weight_mat(num_weights);
    Eigen::Matrix<float, Eigen::Dynamic, 1> bias_mat(num_biases);
    weight_file.read(reinterpret_cast<char*>(weight_mat.data()), num_weights * sizeof(float));
    if (!weight_file.good())
    {
        std::cerr << "Error reading weights! Not enough values!" << std::endl;
        return 1;
    }
    bias_file.read(reinterpret_cast<char*>(bias_mat.data()), num_biases * sizeof(float));
    if (!bias_file.good())
    {
        std::cerr << "Error reading biases! Not enough values!" << std::endl;
        return 1;
    }

	SDFRenderer renderer;
	renderer.setModel(H, N, weight_mat, bias_mat);
	auto img = renderer.render();
	cv::Mat image(img.rows(), img.cols()/3, CV_8UC3, img.data());
	cv::imshow("Render", image);
	cv::waitKey();


	return 0;
}
