#pragma once

struct MeshReference
{
	const Eigen::Ref<const Eigen::MatrixXf> vertices;
	const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> faces;

	MeshReference(const Eigen::Ref<const Eigen::MatrixXf>& _vertices,
		const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>& _faces)
		: vertices(_vertices), faces(_faces) {}

};