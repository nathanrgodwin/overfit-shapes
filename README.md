# Overfit Neural Networks as a Compact Shape Representation
C++/PyTorch Implementation of the Paper: "Overfit Neural Networks as a Compact Shape Representation" with CUDA renderer

[Paper](https://arxiv.org/pdf/2009.09808)

```
@misc{davies2020overfit,
      title={Overfit Neural Networks as a Compact Shape Representation}, 
      author={Thomas Davies and Derek Nowrouzezahrai and Alec Jacobson},
      year={2020},
      eprint={2009.09808},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
```
Render of encoded cube
![Cube Render](/assets/cube.gif)

## Dependencies
### Sampler
- Eigen >= 3
- TBB >= 4
- CGAL >= 4
- Boost >= 1.48

### Renderer
- CUDA >= 9

### Renderer Example
- OpenCV >= 3

### Python Bindings
- pybind11 >= 1.1

## Building
```
mkdir build && cd build && cmake ..
```
Build with your preferred compiler. This was built and tested in Visual Studio 2019 x64.

### Options
- BUILD_RENDERER: Turns renderer project on or off.
- BUILD_PYTHON: Builds python bindings for point sampler and renderer (if BUILD_RENDERER is ON)

### Installing
By default, building the INSTALL project creates the build/workspace directory with all necessary files for examples and training.