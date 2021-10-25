import os
from geomdl import multi
from geomdl import exchange
from geomdl import tessellate
from geomdl.visualization import VisMPL


# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Read file
surf_list = exchange.import_json("shark.json")

# Create Surface container with trim tessellator
surf_cont = multi.SurfaceContainer(surf_list)
surf_cont.set_tessellator(tessellate.TrimTessellate)
surf_cont.delta = 0.01

# Visualize
vis_config = VisMPL.VisConfig()
surf_cont.vis = VisMPL.VisSurface(ctrlpts=False, legend=False, trims=False)
surf_cont.render(evalcolor="steelblue")

# Export as .obj file
# exchange.export_obj(surf_cont, "shark.obj", update_delta=True)
