#pragma once

#include <map>
#include <vector>

#include <Eigen/Core>

#include "ExportSemantics.h"

class EXPORT MeshLoader
{
public:
	MeshLoader();

	static std::pair<Eigen::MatrixXf, Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>
	read(const std::string&);

private:
	template <typename T>
	struct Generic3d
	{
		union {
			T data[3];
			struct
			{
				T x;
				T y;
				T z;
			};
		};

		bool operator<(const Generic3d& other) const
		{
			return std::tie(other.x, other.y, other.z) < std::tie(x, y, z);
		}
	};

public:

	typedef Generic3d<size_t> Face;
	typedef Generic3d<float> Vertex;

protected:

	static bool
	readOBJ(std::istream& is, std::vector<Vertex>&, std::vector<Face>&);

	static bool
	readSTL(std::istream& is, std::vector<Vertex>&, std::vector<Face>&);

	static bool
	readSTLASCII(std::istream& is, std::vector<Vertex>&, std::vector<Face>&);

	static bool
	readSTLBinary(std::istream& is, std::vector<Vertex>&, std::vector<Face>&);

	static bool
	readSTLFace(std::istream& input, std::vector<Vertex>& points, std::vector<Face>& facets,
		int& index, std::map<Vertex, int>& index_map);
};