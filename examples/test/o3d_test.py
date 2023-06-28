import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the .OFF file
mesh = trimesh.load_mesh("../Ducky/duck1.obj")

# Extract vertices and faces
vertices = mesh.vertices
faces = mesh.faces

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Create a Poly3DCollection object for the mesh
poly = Poly3DCollection(vertices[faces], alpha=0.3)

# Set the edge color for the wireframe
poly.set_edgecolor("k")

# Add the Poly3DCollection to the plot
ax.add_collection3d(poly)

# Set axis limits
ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])

# Show the plot
plt.show()
