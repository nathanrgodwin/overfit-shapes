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
    std::cout << num_weights << " weights, " << num_biases << " biases" << std::endl;
    Eigen::Matrix<float, Eigen::Dynamic, 1> weight_mat(num_weights);
    Eigen::Matrix<float, Eigen::Dynamic, 1> bias_mat(num_biases);
    weight_file.read(reinterpret_cast<char*>(weight_mat.data()), num_weights * sizeof(float));
    std::cout << weight_mat[num_weights - 1] << std::endl;
    if (!weight_file.good())
    {
        std::cerr << "Error reading weights! Not enough values!" << std::endl;
        return 1;
    }
    bias_file.read(reinterpret_cast<char*>(bias_mat.data()), num_biases * sizeof(float));
    std::cout << bias_mat[num_biases - 1] << std::endl;
    if (!bias_file.good())
    {
        std::cerr << "Error reading biases! Not enough values!" << std::endl;
        return 1;
    }


	float* nodes = new float[2 * N];
	float* biases = bias_mat.data();
	float* weights = weight_mat.data();
	float3 pos = make_float3(0.57735, 0, 0);

	//fill nodes with bias
	for (unsigned int i = 0; i < N; ++i)
	{
		nodes[i] = biases[i];
	}

	const unsigned int N2 = N * N;
	const unsigned int Hminus = H - 1;

	//Compute nodes from input layer
	for (unsigned int i = 0; i < N; ++i)
	{
		nodes[i] += weights[3 * i] * pos.x + weights[3 * i + 1] * pos.y + weights[3 * i + 2] * pos.z;
		std::cout << weights[3 * i] << ", " << weights[3 * i + 1] << ", " << weights[3 * i + 2] << std::endl;
	}

	unsigned long long weight_offset = 3 * N;

	//Incredibly naive inference
	for (unsigned int layer = 1; layer < H; ++layer)
	{
		unsigned int node_offset = (layer % 2) * N;
		unsigned int prev_node_offset = ((layer - 1) % 2) * N;
		unsigned int layer_weight_offset = (layer - 1) * N2 + weight_offset;
		for (unsigned int node = 0; node < N; ++node)
		{
			nodes[node_offset + node] = biases[node_offset + node];
			for (unsigned int prev_node = 0; prev_node < N; ++prev_node)
			{
				nodes[node_offset + node] += leakyReLU(nodes[prev_node_offset + prev_node], 0.1)
					* weights[layer_weight_offset + prev_node];
			}
		}
	}

	float output = biases[N * H];
	const unsigned int prev_node_offset = (Hminus % 2) * N;
	const unsigned int offset = Hminus * N2 + weight_offset;
	for (unsigned int i = 0; i < N; ++i)
	{
		output += leakyReLU(nodes[prev_node_offset + i], 0.1) * weights[offset + i];
	}

	delete[] nodes;
	output = tanhf(output);
	std::cout << output << std::endl;


	SDFRenderer renderer;
	renderer.setModel(H, N, weight_mat, bias_mat);
	auto img = renderer.render();
	cv::Mat image(img.rows(), img.cols()/3, CV_8UC3, img.data());
	cv::imshow("Render", image);
	cv::waitKey();


	return 0;
}
