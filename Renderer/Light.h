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

	inline std::tuple<float, float, float>
	getColor() const
	{
		return std::make_tuple(color_.x,
				color_.y,
				color_.z);
	}
	__device__ inline const float3
	colorf() const
	{
		return color_;
	}
	inline void
	setColor(std::tuple<float, float, float> color)
	{
		color_ = make_float3(std::get<0>(color), std::get<1>(color), std::get<2>(color));
	}

	inline std::tuple<float,float,float>
	getPosition() const
	{
		return std::make_tuple(pos_.x, pos_.y, pos_.z);
	}
	inline void
	setPosition(std::tuple<float,float,float> pos)
	{
		pos_ = make_float3(std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
	}
	__device__ inline float3
	positionf() const
	{
		return pos_;
	}

	std::tuple<float,float,float>
	getStrength() const
	{
		return std::make_tuple(ambient_strength_, diffuse_strength_, specular_strength_);
	}
	void
	setStrength(std::tuple<float,float,float> strength_val)
	{
		float ambient = std::get<0>(strength_val),
			diffuse = std::get<1>(strength_val),
			specular = std::get<2>(strength_val);
		assert(ambient + diffuse + specular <= 1);
		ambient_strength_ = ambient;
		diffuse_strength_ = diffuse;
		specular_strength_ = specular;
	}

	__host__ __device__ inline float
	ambientStrength() const
	{
		return ambient_strength_;
	}

	__host__ __device__ inline float
	diffuseStrength() const
	{
		return diffuse_strength_;
	}


	__host__ __device__ inline float
	specularStrength() const
	{
		return specular_strength_;
	}

	__host__ __device__ inline unsigned int
	getSpecularPower() const
	{
		return specular_power_;
	}
	inline void
	setSpecularPower(unsigned int power)
	{
		assert(power == 0 || (power & (power - 1)) != 0);
		specular_power_ = power;
	}

};