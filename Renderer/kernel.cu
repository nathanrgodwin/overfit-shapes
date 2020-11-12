#include "kernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "curand_kernel.h"


#include <cmath>
#include <stdio.h>



float __device__ computeSDF(const float3& pos, const
	Renderer::Parameters& params)
{
	return sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z) - 1;
	/*unsigned int N = params.num_layers;
	unsigned int H = params.layer_size;
	float* nodes = new float[2 * N];

	//fill nodes with bias
	for (unsigned int i = 0; i < N; ++i)
	{
		nodes[i] = params.biases[i];
	}

	const unsigned int N2 = N * N;
	const unsigned int Hminus = H - 1;

	//Compute nodes from input layer
	for (unsigned int i = 0; i < N; ++i)
	{
		nodes[i] += params.weights[3 * i] * pos.x + params.weights[3 * i + 1] * pos.y + params.weights[3 * i + 2] * pos.z;
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
			nodes[node_offset + node] = params.biases[node_offset + node];
			for (unsigned int prev_node = 0; prev_node < N; ++prev_node)
			{
				nodes[node_offset + node] += leakyReLU(nodes[prev_node_offset + prev_node], params.slope)
					* params.weights[layer_weight_offset + prev_node];
			}
		}
	}

	float output = params.biases[N * H];
	const unsigned int prev_node_offset = (Hminus % 2) * N;
	const unsigned int offset = Hminus * N2 + weight_offset;
	for (unsigned int i = 0; i < N; ++i)
	{
		output += leakyReLU(nodes[prev_node_offset + i], params.slope) * params.weights[offset + i];
	}

	delete[] nodes;
	output = tanhf(output);
	printf("%f", output);
	return output;*/
}

float3 __device__ objectColor(const float3& pos, const Renderer::Parameters& params)
{
	return make_float3(1.0f);
}

float __device__ distFromOrigin(const float3& position, const float3& direction)
{
	float3 n = normalize(direction);
	float dist = dot(-1 * position, n) / dot(n, n);
	float3 p = position + dist * n;
	return length(p);
}

__device__ float3 rayMarching(const float3& position, const float3& direction, const Renderer::Parameters& params)
{
	float3 color = make_float3(0.0f, 0.0f, 0.0f);

	//For this renderer, all points occupy the unit sphere, so nothing outside needs to be rendered.
	if (dot(direction, -1 * position) < 0 || distFromOrigin(position, direction) > 1) return color;

	float attempted_distance = 0.0f;
	float3 pos = position;
	float3 dir = normalize(direction);

	float remain_dist;
	if ((remain_dist = length(pos)) > 1)
	{
		pos += (remain_dist - 1) * dir;
	}

	while (attempted_distance < params.cam.maxDist())
	{
		float dist = computeSDF(pos, params);

		if (dist < params.min_dist)
		{
			float nx = (computeSDF(make_float3(pos.x + params.eps, pos.y, pos.z), params) - computeSDF(make_float3(pos.x - params.eps, pos.y, pos.z), params));
			float ny = (computeSDF(make_float3(pos.x, pos.y + params.eps, pos.z), params) - computeSDF(make_float3(pos.x, pos.y - params.eps, pos.z), params));
			float nz = (computeSDF(make_float3(pos.x, pos.y, pos.z + params.eps), params) - computeSDF(make_float3(pos.x, pos.y, pos.z - params.eps), params));
			float3 normal = normalize(make_float3(nx, ny, nz));

			//Diffuse lighting
			float3 light_vec = pos - params.light.positionf();
			float light_dot_normal = dot(light_vec, normal) / length(light_vec);
			float diff_angle = std::acosf(light_dot_normal);
			float diff_scale = fmaxf(fminf(1.0f - (fabs(diff_angle - M_PI) / M_PI), 1), 0);
			diff_scale *= params.light.diffuseStrength();

			//Specular lighting
			float3 reflected = light_vec - 2 * light_dot_normal * normal;
			float3 cam_vec = pos - params.cam.positionf();
			float spec_angle = std::acosf(dot(cam_vec, reflected) / (length(reflected) * length(cam_vec)));
			float spec_scale = pow(fmaxf(fminf(1.0f - (fabs(spec_angle - M_PI) / M_PI), 1), 0), (float)params.light.specularPower());
			spec_scale *= params.light.specularStrength();

			color = objectColor(pos, params) * (diff_scale + spec_scale) + params.light.ambientStrength() * params.light.colorf();
			return color;
		}

		attempted_distance += dist;
		pos += dist * dir;
	}

	return color;
}


__global__ void renderImage(Renderer::Parameters params)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= params.width || y >= params.height) return;

	float px = (x / float(params.width) - 0.5f) * 2.0f;
	float py = -(y / float(params.height) - 0.5f) * 2.0f * float(params.height) / float(params.width);
	float3 direction = normalize(params.cam.sidef() * px + params.cam.upf() * py + params.cam.directionf() * params.cam.fovScale());
	float3 color = rayMarching(params.cam.positionf(), direction, params);

	params.device_image[3 * (x + y * params.width) + 0] = fmaxf(fminf(255 * color.x, 255), 0);
	params.device_image[3 * (x + y * params.width) + 1] = fmaxf(fminf(255 * color.y, 255), 0);
	params.device_image[3 * (x + y * params.width) + 2] = fmaxf(fminf(255 * color.z, 255),0);
}

unsigned char*
Renderer::makeImage(unsigned int width, unsigned int height)
{
	unsigned char* deviceImage;
	cudaMalloc(&deviceImage, 3 * width * height);
	return deviceImage;
}

void
Renderer::gpuDelete(unsigned char* image)
{
	cudaFree(image);
}


void
Renderer::gpuDelete(float* data)
{
	cudaFree(data);
}

void
Renderer::render()
{
	assert(params_.weight != nullptr && params_.biases != nullptr);
	dim3 threads(8, 8);
	dim3 blocks(params_.width / threads.x + 1, params_.height / threads.y + 1);
	renderImage<<<blocks, threads>>>(params_);
	cudaMemcpy(params_.image, params_.device_image, 3 * params_.width * params_.height, cudaMemcpyDeviceToHost);
}

void
Renderer::copyDataToGPU(float** dst, const float* src, size_t numel)
{
	cudaMalloc(dst, sizeof(float) * numel);
	cudaMemcpy(*dst, src, sizeof(float) * numel, cudaMemcpyHostToDevice);
}
