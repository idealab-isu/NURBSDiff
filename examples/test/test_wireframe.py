import matplotlib.pyplot as plt
import numpy as np
import random


with open('../../meshes/plane_25.off', 'r') as f:
        lines = f.readlines()

        # skip the first line
        lines = lines[2:]
        lines = random.sample(lines, k=6)
        print(lines)
        # extract vertex positions
        vertex_positions = []
        for line in lines:
            x, y, z = map(float, line.split()[:3])
            vertex_positions.append((x, y, z))
        # range_coord = max(abs(min_coord), abs(max_coord)) / 1
        range_coord = 1
        vertex_positions = [(x, y, z) for x, y, z in vertex_positions]
        target = np.array(vertex_positions).reshape(1, 6, 3)
    ##########################################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



ax.plot_wireframe(target[:, :, 0],
                  target[:, :, 1],
                  target[:, :, 2])

plt.show()