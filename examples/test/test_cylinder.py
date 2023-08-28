import open3d as o3d
import random
import numpy as np

gt_path = "../../meshes/cylinder"
resolution = 10
min_coord = max_coord = 0
# Load the point cloud data
with open(gt_path + '_' + str(resolution * resolution) + '.off', 'r') as f:
    lines = f.readlines()

    # skip the first line
    lines = lines[2:]
    lines = random.sample(lines, k=resolution * resolution)
    # extract vertex positions
    vertex_positions = []
    for line in lines:
        x, y, z = map(float, line.split()[:3])
        min_coord = min(min_coord, x, y, z)
        max_coord = max(max_coord, x, y, z)
        vertex_positions.append((x, y, z))
    range_coord = max(abs(min_coord), abs(max_coord)) / 1
    range_coord = 1
    point_cloud = np.array([(x/range_coord, y/range_coord, z/range_coord) for x, y, z in vertex_positions]).reshape(resolution * resolution, 3)
# point_cloud = o3d.io.read_point_cloud("../../meshes/cylinder_100.off")

def generate_cylinder(point_cloud):
    # Calculate the center of the cylinder
    cylinder_center = np.mean(point_cloud, axis=0)

    # Calculate the radius of the cylinder
    distances = np.linalg.norm(point_cloud - cylinder_center, axis=1)
    cylinder_radius = np.max(distances)

    # Calculate the height of the cylinder
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    return cylinder_center, cylinder_radius, min_z , max_z

# point_cloud = np.random.rand(1000, 3) * 200

cylinder_center, cylinder_radius, min_z , max_z  = generate_cylinder(point_cloud)
 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plotting the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plotting the cylinder
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(min_z, max_z, 10)
u, v = np.meshgrid(u, v)
x = cylinder_center[0] + cylinder_radius * np.cos(u)
y = cylinder_center[1] + cylinder_radius * np.sin(u)
z = v
ax.plot_surface(x, y, z, color='r', alpha=0.2)

plt.show()