/*
	python/pybind11_opencv.hpp: Transparent conversion for OpenCV cv::Mat matrices.
								This header is based on pybind11/eigen.h.

	Copyright (c) 2016 Patrik Huber

	All rights reserved. Use of this source code is governed by a
	BSD-style license that can be found in pybind11's LICENSE file.
*/
#pragma once

#include "pybind11/numpy.h"

#include "Renderer.h"

#include <cstddef>
#include <iostream>

namespace pybind11 {
namespace detail {

template<>
struct type_caster<Image>
{
	bool load(handle src, bool) = delete;

	static handle cast(const Image& src, return_value_policy /* policy */, handle /* parent */)
	{
		return pybind11::array(
			{ (ssize_t)src.rows(), (ssize_t)src.cols() / 3, (ssize_t)3 },
			{
				sizeof(unsigned char) * src.cols(),
				sizeof(unsigned char) * 3,
				sizeof(unsigned char)
			},
			src.data()).release();
	};

	PYBIND11_TYPE_CASTER(Image, _("numpy.ndarray[uint8[m, n, d]]"));
};

}
}