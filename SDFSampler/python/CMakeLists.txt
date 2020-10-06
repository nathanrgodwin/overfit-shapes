pybind11_add_module(pySDFSampler SHARED
	python.cpp)

target_link_libraries(pySDFSampler PUBLIC SDFSampler)

set_target_properties(pySDFSampler PROPERTIES OUTPUT_NAME SDFSampler)