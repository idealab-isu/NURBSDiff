import igl
import numpy as np
import matplotlib.pyplot as plt

# Load a mesh from file.

# ../Ducky/duck1.obj
# ../../meshes/cat.off
vertices, faces = igl.read_triangle_mesh("../../meshes/cat.off")

# Compute the boundary of the mesh.
boundary_loop = igl.boundary_loop(faces)
# print(boundary_loop)
# Map the largest boundary loop to a circle.
mapped_vertices = igl.map_vertices_to_circle(vertices, boundary_loop)
# print(mapped_vertices)
# Perform LSCM parameterization.
_, uv = igl.lscm(vertices, faces, boundary_loop, mapped_vertices)
# def fill_nan_uv(uv, faces):
#     uv_filled = uv.copy()
#     nan_indices = np.isnan(uv).any(axis=1)
#     nan_count = np.sum(nan_indices)

#     while nan_count > 0:
#         for face in faces:
#             for i in range(3):
#                 if nan_indices[face[i]]:
#                     neighbors = [face[(i + 1) % 3], face[(i + 2) % 3]]
#                     if not (nan_indices[neighbors[0]] or nan_indices[neighbors[1]]):
#                         uv_filled[face[i]] = np.mean(uv_filled[neighbors], axis=0)
#                         nan_indices[face[i]] = False
#                         nan_count -= 1

#     return uv_filled

# uv = fill_nan_uv(uv, faces)
for i in range(len(uv)):
    if np.isnan(uv[i][0]):
        uv[i][0] = 0
    if np.isnan(uv[i][1]):
        uv[i][1] = 0
# print(uv)
# Now you can process or save the parameterization as needed.
print(len(uv))
for point in uv:
    print(point[0], point[1], point[0] * point[0] + point[1] * point[1])
# Plot the UV coordinates.
plt.scatter(uv[:, 0], uv[:, 1], marker='.', c='blue')

# Customize the plot.
plt.title("UV Coordinates")
plt.xlabel("U")
plt.ylabel("V")

# Show the plot.
plt.show()


def find_3d_point_from_uv(uv_target, vertices, faces, uv):
    def normalize_uv(uv):
        min_uv = np.min(uv, axis=0)
        max_uv = np.max(uv, axis=0)
        range_uv = max_uv - min_uv
        normalized_uv = 2 * (uv - min_uv) / range_uv - 1
        return normalized_uv

    def barycentric_coordinates(uv_target, uv_triangle):
        v0, v1, v2 = uv_triangle
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = uv_target - v0

        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)

        denominator = d00 * d11 - d01 * d01
        alpha = (d11 * d20 - d01 * d21) / denominator
        beta = (d00 * d21 - d01 * d20) / denominator
        gamma = 1 - alpha - beta

        return np.array([alpha, beta, gamma])

    normalized_uv = normalize_uv(uv)
    closest_point_3d = None
    min_distance = float('inf')

    for face in faces:
        uv_triangle = np.array([normalized_uv[face[0]], normalized_uv[face[1]], normalized_uv[face[2]]])
        barycentric_coords = barycentric_coordinates(uv_target, uv_triangle)

        if all(-1e-6 <= coord <= 1 + 1e-6 for coord in barycentric_coords):
            vertex_triangle = np.array([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
            point_3d = barycentric_coords @ vertex_triangle
            distance = np.linalg.norm(uv_target - np.dot(barycentric_coords, uv_triangle))

            if distance < min_distance:
                min_distance = distance
                closest_point_3d = point_3d

    if closest_point_3d is not None:
        return closest_point_3d
    else:
        raise ValueError("The target UV coordinate is not inside any triangle.")

# Example usage:
print(uv.shape)

# uv_target = []
# for k in range(1, 11):
#     j = k / 10
#     for i in range(0, 11):
#         x = j * i / 10
#         y = np.sqrt(j - x)
#         uv_target.append([x, y])
#         uv_target.append([-x, y])
#         uv_target.append([x, -y])
#         uv_target.append([-x, -y])
    
# uv_target = np.array(uv_target)
uv_target = uv.copy()
with open(f'uvmapping/duck.OFF', 'w') as f:
            # Loop over the array rows
            f.write('OFF\n')
            f.write(str(uv_target.shape[0]) + ' ' + '0 0\n')
for point in uv_target:
    try:
        point_3d = find_3d_point_from_uv(point, vertices, faces, uv)
        print(f"The 3D point corresponding to UV coordinate {point} is: {point_3d}")

        with open(f'uvmapping/duck.OFF', 'a') as f:
    
                # print(predicted_target[i, j, :])
                line = str(point_3d[0]) + ' ' + str(point_3d[1]) + ' ' + str(point_3d[2]) + '\n'
                f.write(line)
    except:
        pass
# uv_target = np.array([0.9, 0.01])
# point_3d = find_3d_point_from_uv(uv_target, vertices, faces, uv)
# print(f"The 3D point corresponding to UV coordinate {uv_target} is: {point_3d}")
