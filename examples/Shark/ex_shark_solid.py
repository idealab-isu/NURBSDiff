import os
from geomdl import multi
from geomdl import exchange
from geomdl import tessellate
from geomdl.visualization import VisVTK


# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Read file
surf_list = exchange.import_json("shark_solid.json")

# Create Surface container with trim tessellator
surf_cont = multi.SurfaceContainer(surf_list)
surf_cont.set_tessellator(tessellate.TrimTessellate)
surf_cont.sample_size = 100

# Visualize
vis_config = VisVTK.VisConfig(ctrlpts=False, legend=False, trims=False)
surf_cont.vis = VisVTK.VisSurface(vis_config)
surf_cont.render(evalcolor="grey")

# Export as .obj file
# exchange.export_obj(surf_cont, "shark_solid.obj", update_delta=True)
