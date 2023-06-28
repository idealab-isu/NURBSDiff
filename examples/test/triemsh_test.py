import trimesh
import numpy as np

# Load the .OFF file
mesh = trimesh.load_mesh("../Ducky/duck1.obj")

# Unwrap the UV coordinates using xatlas
unwrapped_mesh = mesh.unwrap()

# Visualize the unwrapped mesh with UV mapping
unwrapped_mesh.show()

