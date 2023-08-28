import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_cube(point_cloud):
    # Calculate the minimum and maximum coordinates of the point cloud
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)

    # Calculate the center point of the cube
    cube_center = (max_coords + min_coords) / 2.0

    # Calculate the size (edge length) of the cube
    cube_size = np.max(max_coords - min_coords)

    # Calculate the vertices of the cube
    cube_vertices = np.array([
        [cube_center[0] - cube_size / 2, cube_center[1] - cube_size / 2, cube_center[2] - cube_size / 2],
        [cube_center[0] + cube_size / 2, cube_center[1] - cube_size / 2, cube_center[2] - cube_size / 2],
        [cube_center[0] + cube_size / 2, cube_center[1] + cube_size / 2, cube_center[2] - cube_size / 2],
        [cube_center[0] - cube_size / 2, cube_center[1] + cube_size / 2, cube_center[2] - cube_size / 2],
        [cube_center[0] - cube_size / 2, cube_center[1] - cube_size / 2, cube_center[2] + cube_size / 2],
        [cube_center[0] + cube_size / 2, cube_center[1] - cube_size / 2, cube_center[2] + cube_size / 2],
        [cube_center[0] + cube_size / 2, cube_center[1] + cube_size / 2, cube_center[2] + cube_size / 2],
        [cube_center[0] - cube_size / 2, cube_center[1] + cube_size / 2, cube_center[2] + cube_size / 2]
    ])

    return cube_vertices

# Example point cloud represented as a NumPy array
point_cloud = np.random.rand(100, 3)

# Generate the cube
cube_vertices = generate_cube(point_cloud)

# Plotting the point cloud and the cube
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o')
ax.plot_trisurf(cube_vertices[:, 0], cube_vertices[:, 1], cube_vertices[:, 2], color='r', alpha=0.2)

# Setting the plot limits
ax.set_xlim3d(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]))
ax.set_ylim3d(np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1]))
ax.set_zlim3d(np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2]))

# Labeling the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Displaying the plot
plt.show()
