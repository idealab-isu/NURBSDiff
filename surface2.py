#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Examples for the NURBS-Python Package
    Released under MIT License
    Developed by Onur Rauf Bingol (c) 2016-2018
"""

import os
from geomdl import BSpline
from geomdl import exchange
from geomdl.visualization import VisMPL as vis


# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Create a BSpline surface instance
surf = BSpline.Surface()

# Set degrees
surf.degree_u = 4
surf.degree_v = 4

# Set control points
surf.set_ctrlpts(*exchange.import_txt("u_test.cpt", sep=',', two_dimensional=False))
# Set knot vectors
with open('u_test.knotu', 'r') as f:
    surf.knotvector_u = [float(x) for x in f.read().split()]
surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]

# Set evaluation delta
surf.delta = 0.025

# Evaluate surface points
surf.evaluate()

# Import and use Matplotlib's colormaps
from matplotlib import cm

# Plot the control point grid and the evaluated surface
vis_comp = vis.VisSurface()
# Set the size of the control points to 10 pixels
# vis_comp.ctrlpts_size = 10
vis_comp.ctrlpts_offset = 50
surf.vis = vis_comp
surf.render(colormap=cm.cool, cpcolor='red')
exchange.export_obj(surf, "u_test.obj")
# Good to have something here to put a breakpoint
pass
