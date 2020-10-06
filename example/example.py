from SDFSampler import PointSampler
import numpy as np

vertices = np.array([[0.0, 0.0, 0.0],
        [0.0,  0.0,  1.0],
        [0.0,  1.0,  0.0],
        [0.0,  1.0,  1.0],
        [1.0,  0.0,  0.0],
        [1.0,  0.0,  1.0],
        [1.0,  1.0,  0.0],
        [1.0,  1.0,  1.0]], np.dtype(float))

faces = np.array([[0, 6, 4],
        [0, 2, 6],
        [0, 3, 2],
        [0, 1, 3],
        [2, 7, 6],
        [2, 3, 7],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 7],
        [1, 7, 3]])

sampler = PointSampler(vertices, faces)
print(sampler.sample())