import numpy as np
import random
import matplotlib.pyplot as plt

resolution_u = 15
resolution_v = 15
min_coord = max_coord = 0
# with open('generated/ducky/duck1'+ '.ctrlpts', 'r') as f:
with open('../../meshes/unsupervised_predicted_ctrpts_ctrpts_15x15_eval_30_reconstruct_32x32'+ '.OFF', 'r') as f:
    lines = f.readlines()

    # skip the first line
    lines = lines[2:]
    lines = lines[:resolution_u * resolution_v]
    # lines = random.sample(lines, k=resolution_u * resolution_v)
    # extract vertex positions
    vertex_positions = []
    for line in lines:
        x, y, z = map(float, line.split()[:3])
        min_coord = min(min_coord, x, y, z)
        max_coord = max(max_coord, x, y, z)
        vertex_positions.append((x, y, z))
    # range_coord = max(abs(min_coord), abs(max_coord))
    range_coord = 1
    vertex_positions = np.array([(x/range_coord, y/range_coord, z/range_coord) for x, y, z in vertex_positions]).reshape(resolution_u, resolution_v, 3)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.azim = -90
    ax.dist = 6.5
    ax.elev = 30
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    ax.plot_wireframe(vertex_positions[:,:,0], vertex_positions[:,:,1], vertex_positions[:,:,2])

    plt.show()
