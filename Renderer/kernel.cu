#include "kernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "curand_kernel.h"


#include <cmath>
#include <stdio.h>


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
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float3 val = make_float3(1, 0, 0);

	int M = params.N, K = 3;

	bool relu = true;

	float* weights = params.weights, * biases = params.biases;

	buffer[0] = pos.x;
	buffer[1] = pos.y;
	buffer[2] = pos.z;
	float* output_buffer;
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
		/*if (x == params.width / 2 && y == params.height / 2)
		{
			printf("---------------- Weights %d -----------------\n", l);
			for (int i = 0; i < M; ++i)
			{
				for (int j = 0; j < K; ++j)
				{
					printf("%f, ", weights[i * K + j]);
				}
				printf("\n");
			}
		}*/

		weights += M * K;
		biases += M;

		K = M;
	}

	float output = std::tanh(output_buffer[0]);
	//if (x == params.width/2 && y == params.height/2) printf("%f\n", output);
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

__device__ float3 rayMarching(const float3& position, const float3& direction, const Renderer::Parameters& params)
{
	float* buffer = new float[2 * params.N];
	float3 color = params.background_color;

	//For this renderer, all points occupy the unit sphere, so nothing outside needs to be rendered.
	if (dot(direction, -1 * position) < 0 || distFromOrigin(position, direction) > 1)
	{
		delete[] buffer;
		return color;
	}

	float attempted_distance = 0.0f;
	float3 pos = position;
	float3 dir = normalize(direction);

	float remain_dist;
	if ((remain_dist = length(pos)) > 1)
	{
		pos += (remain_dist - 1) * dir;
	}

	while (attempted_distance < params.cam.getMaxDist())
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
			delete[] buffer;
			return color;
		}

		attempted_distance += dist;
		pos += dist * dir;
	}

	delete[] buffer;
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
