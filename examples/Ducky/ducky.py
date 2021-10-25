#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Plotting the Duck Example from GLScene with NURBS-Python

    GLScene: http://wiki.freepascal.org/GLScene
    GLScene on GitHub: https://github.com/cutec-chris/GLScene

    The "Ducky" example contains 3 surfaces. The surface data is accessible via the "GLScene on GitHub" link above in
    the directory "Demos/media/" with the file names "duck1.nurbs", "duck2.nurbs" and "duck3.nurbs". The contents of
    these files are extracted into NURBS-Python format.

    The main difference between the surface formats is the row order. Most of the time it affects the evaluation result.
    It is easy to understand the issues caused by the row order: Using a 1-dimensional array of control points in the
    wrong order may cause a "hole" or leave some regions unevaluated. For the 2-dimensional array of control points,
    most of the time you will get an IndexError exception.

    NURBS-Python uses v-row order but GLScene uses u-row order (please see docs for more details on the row order).
    This means that u- and v-directions of the *.nurbs files must be transposed (or flipped) in order to get the
    correct shapes. This example fixes the issues caused by the row order difference.
"""

from geomdl import NURBS
from geomdl import multi
from geomdl import exchange
from geomdl import compatibility
from geomdl.visualization import VisMPL


def read_weights(filename, sep=","):
    try:
        with open(filename, "r") as fp:
            content = fp.read()
            content_arr = [float(w) for w in (''.join(content.split())).split(sep)]
            return content_arr
    except IOError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise e


# duck1.nurbs (body)

# Process control points and weights
d2_ctrlpts = exchange.import_txt("duck1.ctrlpts", separator=" ")
d1_weights = read_weights("duck1.weights")
d1_ctrlptsw = compatibility.combine_ctrlpts_weights(d2_ctrlpts, d1_weights)

# Create a NURBS surface
duck1 = NURBS.Surface()
duck1.order_u = 4
duck1.order_v = 4
duck1.ctrlpts_size_u = 14
duck1.ctrlpts_size_v = 13
duck1.ctrlptsw = d1_ctrlptsw
duck1.knotvector_u = [-1.5708, -1.5708, -1.5708, -1.5708, -1.0472, -0.523599, 0, 0.523599, 0.808217,
                      1.04015, 1.0472, 1.24824, 1.29714, 1.46148, 1.5708, 1.5708, 1.5708, 1.5708]
duck1.knotvector_v = [-3.14159, -3.14159, -3.14159, -3.14159, -2.61799, -2.0944, -1.0472, -0.523599,
                      6.66134e-016, 0.523599, 1.0472, 2.0944, 2.61799, 3.14159, 3.14159, 3.14159, 3.14159]

#############################################
# DERIVATIVES AT THE NODES OF THE TRIANGLES #  - Start
#############################################

# Tessellate
duck1.tessellate()

# Get vertices
vertex_array = duck1.tessellator.vertices

# Get triangles
triangle_array = duck1.tessellator.faces

# Loop through triangles
for tri in triangle_array:
    verts = tri.vertices
    for v in verts:
        vertex_id = v.id  # an arbitrary number used for vertex indexing
        # vertex_coords = v.data  # Cartesian coordinates of the vertex
        vertex_uv = v.uv
        skl = duck1.derivatives(vertex_uv[0], vertex_uv[1], order=2)
        skl[0][0]  # s  --> original surface point
        skl[0][1]  # ds/dv
        skl[1][0]  # ds/du
        skl[1][1]  # ds2/dvdu
        skl[0][2]  # ds2/d2v
        skl[2][0]  # ds2/d2u
        skl[2][1]  # ds3/d2udv  --> optional, only for showing, do not use

# Export as Wavefront .OBJ file
exchange.export_obj(duck1, "duck1.obj")

#############################################
# DERIVATIVES AT THE NODES OF THE TRIANGLES #  - End
#############################################

# duck2.nurbs (top beak)

# Process control points and weights
d2_ctrlpts = exchange.import_txt("duck2.ctrlpts", separator=" ")

# Create a NURBS surface
duck2 = NURBS.Surface()
duck2.order_u = 4
duck2.order_v = 4
duck2.ctrlpts_size_u = 9
duck2.ctrlpts_size_v = 10
duck2.ctrlpts = d2_ctrlpts
duck2.knotvector_u = [0, 0, 0, 0, 0.145456, 0.265731, 0.436096, 0.583258, 0.847704, 1, 1, 1, 1]
duck2.knotvector_v = [0, 0, 0, 0, 0.179541, 0.317924, 0.485586, 0.507528, 0.709398, 0.813231, 1, 1, 1, 1]


# duck3.nurbs (bottom beak)

# Process control points and weights
d3_ctrlpts = exchange.import_txt("duck3.ctrlpts", separator=" ")

# Create a NURBS surface
duck3 = NURBS.Surface()
duck3.order_u = 4
duck3.order_v = 4
duck3.ctrlpts_size_u = 6
duck3.ctrlpts_size_v = 6
duck3.ctrlpts = d3_ctrlpts
duck3.knotvector_u = [0, 0, 0, 0, 0.333333, 0.666667, 1, 1, 1, 1]
duck3.knotvector_v = [0, 0, 0, 0, 0.333333, 0.666667, 1, 1, 1, 1]

# # Plotting
ducky = multi.SurfaceContainer(duck1, duck2, duck3)
vis_config = VisMPL.VisConfig(ctrlpts=False, wireframe=True, surface=False)

# Use Matplotlib's colormaps to color the duck
from matplotlib import cm

ducky.vis = VisMPL.VisSurface(vis_config)
ducky.sample_size = 50
ducky.render(colormap=[cm.Wistia, cm.copper, cm.copper])

pass
