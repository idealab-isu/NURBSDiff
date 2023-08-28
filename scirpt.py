import numpy as np
import random
resolution = 64
obj = 'horse'
with open('./meshes/' + obj + '_n.off', 'r') as f:
    lines = f.readlines()

    # skip the first line
    lines = lines[2:]
    lines = random.sample(lines, k=resolution * resolution)
    # extract vertex positions
    vertex_positions = []
    min_coord = max_coord = 0
    for line in lines:
        x, y, z = map(float, line.split()[:3])
        min_coord = min(min_coord, x, y, z)
        max_coord = max(max_coord, x, y, z)
        vertex_positions.append((x, y, z))
    range_coord = max(abs(min_coord), abs(max_coord)) / 1
    # range_coord = 1
    vertex_positions = np.array([(x/range_coord, y/range_coord, z/range_coord) for x, y, z in vertex_positions]).reshape(-1, 3)
        

point_cloud = np.save('./examples/test/generated/npy/' + obj + '_n' + '.xyz.npy', vertex_positions)

# x = np.load('meshes/horse.xyz.npy')
# print(x.shape)
# print(x)