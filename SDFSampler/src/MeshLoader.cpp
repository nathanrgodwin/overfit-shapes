#include "IO/MeshLoader.h"

#include "FilesystemSelection.h"

#include <fstream>
#include <iostream>
#include <map>

MeshLoader::MeshLoader()
{}

std::pair<Eigen::MatrixXf, Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>
MeshLoader::read(const std::string& file)
{
    Eigen::MatrixXf vertices;
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> faces;

    std::vector<Vertex> vertex_vec;
    std::vector<Face> face_vec;

	auto extension = fs::path(file).extension();

	std::ifstream is(file, std::ios_base::in | std::ios_base::binary);
	if (!is.is_open())
	{
        return std::make_pair(vertices, faces);
	}
    bool result = false;
	if (extension == ".stl" || extension == ".STL")
	{
		result = readSTL(is, vertex_vec, face_vec);
	}
	else if (extension == ".obj" || extension == ".OBJ")
	{
		result = readOBJ(is, vertex_vec, face_vec);
	}
	else
	{
		is.close();
	}
    if (result)
    {
        vertices.resize(vertex_vec.size(), 3);
        faces.resize(face_vec.size(), 3);

        for (size_t i = 0; i < vertex_vec.size(); ++i)
        {
            vertices(i, 0) = vertex_vec[i].x;
            vertices(i, 1) = vertex_vec[i].y;
            vertices(i, 2) = vertex_vec[i].z;
        }

        for (size_t i = 0; i < face_vec.size(); ++i)
        {
            faces(i, 0) = face_vec[i].x;
            faces(i, 1) = face_vec[i].y;
            faces(i, 2) = face_vec[i].z;
        }
    }
	return std::make_pair(vertices, faces);
}

bool
MeshLoader::readOBJ(std::istream& input, std::vector<Vertex>& points, std::vector<Face>& faces)
{
    Vertex v;
    std::string line;
    while (getline(input, line))
    {
        if (line[0] == 'v' && line[1] == ' ')
        {
            std::istringstream iss(line.substr(1));
            if (!(iss >> v)) return false;
            points.push_back(v);
        }
        else if (line[0] == 'f')
        {
            Face f;
            std::istringstream iss(line.substr(1));
            if (!(iss >> f)) return false;
            for (int i = 0; i < 3; ++i)
            {
                if (f.data[i] < 1)
                {
                    f.data[i] += points.size();
                }
                else
                {
                    --f.data[i];
                }
            }
            faces.push_back(f);
        }
        else
        {
            continue;
        }
    }
    return true;
}

bool
MeshLoader::readSTLFace(std::istream& input,
    std::vector<MeshLoader::Vertex>& points,
    std::vector<MeshLoader::Face>& facets,
    int& index,
    std::map<MeshLoader::Vertex, int>& index_map)
{
    std::string s;
    std::string vertex("vertex"), endfacet("endfacet");

    int count = 0;
    Vertex p;
    Face facet;

    while (input >> s)
    {
        if (s == endfacet)
        {
            if (count != 3) return false;

            facets.push_back(facet);
            return true;
        }
        else if (s == vertex)
        {
            if (count >= 3) return false;

            if (!(input >> p).good())
            {
                return false;
            }
            else
            {
                typename std::map<Vertex, int>::iterator iti = index_map.insert(std::make_pair(p, -1)).first;

                if (iti->second == -1)
                {
                    facet.data[count] = index;
                    iti->second = index++;
                    points.push_back(p);
                }
                else
                {
                    facet.data[count] = iti->second;
                }
            }

            ++count;
        }
    }

    return false;
}


bool
MeshLoader::readSTLASCII(std::istream& input, std::vector<Vertex>& points, std::vector<Face>& faces)
{

    if (!input.good()) return true;

    int index = 0;
    std::map<Vertex, int> index_map;

    std::string s, facet("facet"), endsolid("endsolid");

    while (input >> s)
    {
        if (s == facet)
        {
            if (!readSTLFace(input, points, faces, index, index_map)) return false;
        }
        else if (s == endsolid)
        {
           return true;
        }
    }

    return false;
}


bool
MeshLoader::readSTLBinary(std::istream& input, std::vector<Vertex>& points, std::vector<Face>& faces)
{
    input.clear();
    input.seekg(0, std::ios::beg);

    if (!input.good()) return true;

    int pos = 0;
    char c;

    while (pos < 80)
    {
        input.read(reinterpret_cast<char*>(&c), sizeof(c));
        if (!input.good()) break;

        ++pos;
    }

    if (pos != 80) return true;

    int index = 0;
    std::map<Vertex, int> index_map;

    uint32_t num_faces;
    if (!(input.read(reinterpret_cast<char*>(&num_faces), sizeof(num_faces)))) return false;

    for (uint32_t i = 0; i < num_faces; ++i)
    {
        float normal[3];
        if (!(input.read(reinterpret_cast<char*>(&normal[0]), sizeof(normal[0]))) ||
            !(input.read(reinterpret_cast<char*>(&normal[1]), sizeof(normal[1]))) ||
            !(input.read(reinterpret_cast<char*>(&normal[2]), sizeof(normal[2]))))
        {
            std::cerr << "Malformed normal" << std::endl;
            return false;
        }

        Face facet;
        for (int j = 0; j < 3; ++j)
        {
            Vertex v;
            if (!(input.read(reinterpret_cast<char*>(&v.x), sizeof(float))) ||
                !(input.read(reinterpret_cast<char*>(&v.y), sizeof(float))) ||
                !(input.read(reinterpret_cast<char*>(&v.z), sizeof(float))))
            {
                std::cerr << "Malformed vertex." << input.fail() << ", " << input.good() << ", " << input.bad() << ", " << input.eof() << std::endl;
                return false;
            }

            typename std::map<Vertex, int>::iterator iti = index_map.insert(std::make_pair(v, -1)).first;

            if (iti->second == -1)
            {
                facet.data[j] = index;
                iti->second = index++;
                points.push_back(v);
            }
            else
            {
                facet.data[j] = iti->second;
            }
        }

        faces.push_back(facet);

        char c;
        if (!(input.read(reinterpret_cast<char*>(&c), sizeof(c))) ||
            !(input.read(reinterpret_cast<char*>(&c), sizeof(c))))
        {
            std::cerr << "Malformed attribute" << std::endl;
            return false;
        }
    }

    return true;
}


bool
MeshLoader::readSTL(std::istream& is, std::vector<Vertex>& points, std::vector<Face>& faces)
{
    int pos = 0;

    unsigned char c;

    while (is.read(reinterpret_cast<char*>(&c), sizeof(c)))
    {
        if (!isspace(c))
        {
            is.unget();
            break;
        }
        ++pos;
    }

    if (!is.good()) return true;

    if (pos > 80) return readSTLASCII(is, points, faces);

    std::string s, solid("solid");

    char word[5];
    if (is.read(reinterpret_cast<char*>(&word[0]), sizeof(c)) &&
        is.read(reinterpret_cast<char*>(&word[1]), sizeof(c)) &&
        is.read(reinterpret_cast<char*>(&word[2]), sizeof(c)) &&
        is.read(reinterpret_cast<char*>(&word[3]), sizeof(c)) &&
        is.read(reinterpret_cast<char*>(&word[4]), sizeof(c)))
    {
        s = std::string(word, 5);
        pos += 5;
    }
    else
    {
        return true;
    }

      // If the first word is not 'solid', the file must be binary
    if (s != solid)
    {
        if (readSTLBinary(is, points, faces))
        {
            return true;
        }
        else
        {
            is.clear();
            is.seekg(0, std::ios::beg);
            return readSTLASCII(is, points, faces);
        }
    }

    // Now, we have found the keyword "solid" which is supposed to indicate that the file is ASCII
    if (readSTLASCII(is, points, faces))
    {
        // correctly read the input as an ASCII file
        return true;
    }
    else // Failed to read the ASCII file
    {
        // It might have actually have been a binary file... ?
        return readSTLBinary(is, points, faces);
    }
}