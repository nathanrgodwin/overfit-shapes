#include "kernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "curand_kernel.h"


#include <cmath>
#include <stdio.h>
#include <stdexcept>
#include <sstream>

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
		std::stringstream ss;
		ss << "CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
		if (abort) throw std::runtime_error(ss.str());
	}
}

__device__ float
leakyReLU(float val)
{
	return fmaxf(0, val) + 0.1 * fminf(0, val);
}

__device__ void
matvecmul(bool relu, int M, int K, const float* a, const float* b, const float* c, float* d)
{
	for (int i = 0; i < M; ++i)
	{
		float val = 0;
		for (int j = 0; j < K; ++j)
		{
			val += a[i * K + j] * b[j];
		}
		val += c[i];
		d[i] = (relu) ? leakyReLU(val) : val;
	}
}

float __device__ computeSDF(const float3& pos, float * buffer, const
	Renderer::Parameters& params)
{
	float3 val = make_float3(1, 0, 0);

	int M = params.N, K = 3;

	bool relu = true;

	float* weights = params.weights, * biases = params.biases;

	buffer[0] = pos.x;
	buffer[1] = pos.y;
	buffer[2] = pos.z;
	float* output_buffer = buffer;
	for (int l = 0; l < params.H + 1; ++l)
	{
		float* input_buffer = buffer + (M * (l % 2));
		output_buffer = buffer + (M * ((l + 1) % 2));
		if (l == params.H)
		{
			M = 1;
			relu = false;
		}

		matvecmul(relu, M, K, weights, input_buffer, biases, output_buffer);

		weights += M * K;
		biases += M;

		K = M;
	}

	float output = std::tanh(output_buffer[0]);
	return output;
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

float __device__ distToSphere(const float3& position, const float3& direction)
{
	float a = dot(direction, direction);
	float b = 2.0 * dot(position, direction);
	float c = dot(position, position) - 1;
	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0.0)
	{
		return -1;
	}
	else
	{
		float numerator = -b - sqrt(discriminant);
		if (numerator > 0)
		{
			return numerator / (2.0 * a);
		}

		numerator = -b + sqrt(discriminant);
		if (numerator > 0)
		{
			return numerator / (2.0 * a);
		}
		else
		{
			return -1;
		}
	}
}

#define EPS 0.000001

__device__ float3 rayMarching(const float3& position, const float3& direction, const Renderer::Parameters& params)
{
	extern __shared__ float shared_mem[];
	size_t offset = 2 * params.N * (threadIdx.x + threadIdx.y * blockDim.x);
	float* buffer = shared_mem + offset;
	float3 color = params.background_color;

	float3 pos = position;
	float3 dir = normalize(direction);
	float intersect_dist = distToSphere(pos, dir);

	//For this renderer, all points occupy the unit sphere, so nothing outside needs to be rendered.
	if (intersect_dist < 0)
	{
		return color;
	}
	else
	{
		pos += (intersect_dist + EPS) * dir;
	}

	float pos_len = length(pos);
	while (pos_len < 1)
	{

		float dist = computeSDF(pos, buffer, params);

		if (dist < params.min_dist)
		{
			float nx = (computeSDF(make_float3(pos.x + params.eps, pos.y, pos.z), buffer, params) - computeSDF(make_float3(pos.x - params.eps, pos.y, pos.z), buffer, params));
			float ny = (computeSDF(make_float3(pos.x, pos.y + params.eps, pos.z), buffer, params) - computeSDF(make_float3(pos.x, pos.y - params.eps, pos.z), buffer, params));
			float nz = (computeSDF(make_float3(pos.x, pos.y, pos.z + params.eps), buffer, params) - computeSDF(make_float3(pos.x, pos.y, pos.z - params.eps), buffer, params));
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
			float spec_scale = pow(fmaxf(fminf(1.0f - (fabs(spec_angle - M_PI) / M_PI), 1), 0), (float)params.light.getSpecularPower());
			spec_scale *= params.light.specularStrength();

			color = objectColor(pos, params) *(diff_scale + spec_scale) + params.light.ambientStrength() * params.light.colorf();
			return color;
		}

		pos += dist * dir;
		pos_len = length(pos);
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
	cudaCheck(cudaMalloc(&deviceImage, 3 * width * height));
	return deviceImage;
}

void
Renderer::gpuDelete(unsigned char* image)
{
	cudaCheck(cudaFree(image));
}


void
Renderer::gpuDelete(float* data)
{
	cudaCheck(cudaFree(data));
}

void
Renderer::render()
{
	dim3 block_size((params_.N <= 64) ? 8 : 4, 8);
	dim3 grid_size(params_.width / block_size.x + 1, params_.height / block_size.y + 1);
	renderImage<<<grid_size, block_size, 2 * block_size.x * block_size.y * params_.N * sizeof(float)>>>(params_);
	cudaCheck(cudaPeekAtLastError());
	cudaCheck(cudaMemcpy(params_.image, params_.device_image, 3 * params_.width * params_.height, cudaMemcpyDeviceToHost));
}

void
Renderer::copyDataToGPU(float** dst, const float* src, size_t numel)
{
	cudaCheck(cudaMalloc(dst, sizeof(float) * numel));
	cudaCheck(cudaMemcpy(*dst, src, sizeof(float) * numel, cudaMemcpyHostToDevice));
}
