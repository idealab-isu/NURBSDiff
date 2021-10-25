import os
from geomdl import multi
from geomdl import exchange
from geomdl import tessellate
from geomdl import trimming
from geomdl.visualization import VisVTK


if __name__ == "__main__":
    # Fix file path
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Read file
    surf_list = exchange.import_json("hammer.json")

    # Create Surface container
    surf_cont = multi.SurfaceContainer(surf_list)

    # Fix trim curves
    surf_cont = trimming.fix_multi_trim_curves(surf_cont, delta=0.05)

    # Set sample size
    surf_cont.sample_size = 10

    # Set trim tessellator for all surfaces
    surf_cont.tessellator = tessellate.TrimTessellate()
    # surf_cont.tessellate(num_procs=8)
    # surf_cont.set_tessellator(tessellate.TrimTessellate)

    # Visualize
    surf_cont.vis = VisVTK.VisSurface()
    surf_cont.render(evalcolor="grey", num_procs=16)

    # Export as .obj file
    # exchange.export_obj(surf_cont, "hammer.obj", update_delta=True)
