from OverfitShapes import PointSampler, MeshLoader, normalizeMeshToUnitSphere
import numpy as np
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.pyplot as plt

def main(plot_mesh = False):
    vertices, faces = MeshLoader.read("cube.obj")
    normalizeMeshToUnitSphere(vertices, faces)

    sampler = PointSampler(vertices, faces)
    boundary_points = sampler.sample(int(0.99*1e3), 10)

    # Testing indicated very poor SDF accuracy outside the mesh boundary which complicated
    # raymarching operations.
    # Adding samples through the unit sphere improves accuracy farther from the boundary,
    # but still within the unit sphere
    general_points = sampler.sample(int(0.01*1e3), 1)
    pts = (np.concatenate((boundary_points[0], general_points[0])),
                np.concatenate((boundary_points[1], general_points[1])))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)


    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color="r", alpha=0.1)


    ax.scatter(pts[0][:,0], pts[0][:,1], pts[0][:,2])

    if (plot_mesh):
        n = faces.shape[0]
        for i in range(n):
            for j in range(3):
                if (j == 2):
                    pts = np.vstack((vertices[faces[i,j],:], vertices[faces[i,0],:]))
                else:
                    pts = np.vstack((vertices[faces[i,j],:], vertices[faces[i,j+1],:]))
                ax.plot(pts[:,0], pts[:,1], pts[:,2], color='black')

    plt.show()

if __name__ == "__main__":
    main()