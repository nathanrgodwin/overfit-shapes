#pragma once

#include <cassert>
#include <tuple>
#include "cuda_runtime.h"

class Light
{
private:
	float3 color_;
	float3 pos_;
	float diffuse_strength_;
	float specular_strength_;
	float ambient_strength_;
	unsigned int specular_power_;

public:
	Light() :
		specular_power_(8),
		diffuse_strength_(0.65),
		specular_strength_(0.25),
		ambient_strength_(0.1),
		color_(make_float3(1, 1, 1)),
		pos_(make_float3(1, 1, 1)) {}

	inline std::tuple<unsigned char, unsigned char, unsigned char>
	color() const
	{
		return std::make_tuple((unsigned char)(255 * color_.x),
				(unsigned char)(255 * color_.y),
				(unsigned char)(255 * color_.z));
	}
	__device__ inline const float3
	colorf() const
	{
		return color_;
	}
	inline void
	color(unsigned char r, unsigned char g, unsigned char b)
	{
		color_ = make_float3(r, g, b) / 255.0f;
	}

	inline std::tuple<float,float,float>
	position() const
	{
		return std::make_tuple(pos_.x, pos_.y, pos_.z);
	}
	inline void
	position(float x, float y, float z)
	{
		pos_ = make_float3(x, y, z);
	}
	__device__ inline float3
	positionf() const
	{
		return pos_;
	}

	std::tuple<float,float,float>
	strength() const
	{
		return std::make_tuple(ambient_strength_, diffuse_strength_, specular_strength_);
	}
	void
	strength(float ambient, float diffuse, float specular)
	{
		assert(ambient + diffuse + specular <= 1);
		ambient_strength_ = ambient;
		diffuse_strength_ = diffuse;
		specular_strength_ = specular;
	}

	__device__ inline float
	ambientStrength() const
	{
		return ambient_strength_;
	}

	__device__ inline float
	diffuseStrength() const
	{
		return diffuse_strength_;
	}


	__device__ inline float
	specularStrength() const
	{
		return specular_strength_;
	}

	__device__ inline unsigned int
	specularPower() const
	{
		return specular_power_;
	}
	inline void
	specularPower(unsigned int power)
	{
		assert(power == 0 || (power & (power - 1)) != 0);
		specular_power_ = power;
	}

};