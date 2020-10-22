# Overfit Neural Networks as a Compact Shape Representation
C++/PyTorch Implementation of the Paper: "Overfit Neural Networks as a Compact Shape Representation"

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

## Dependencies
- Eigen >= 3
- TBB >= 4

## Building
```
mkdir build && cd build && cmake ..
```
Build with your preferred compiler. This was built and tested in Visual Studio 2019 x64.

### SDFSampler
Builds a C++ library with Python bindings for point selection.