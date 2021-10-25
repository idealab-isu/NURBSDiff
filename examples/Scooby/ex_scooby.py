import os
from geomdl import multi, exchange, tessellate, trimming
from geomdl.visualization import VisVTK


# Required for multiprocessing (Windows only)
if __name__ == "__main__":
    # Fix file path
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Read file
    surf_list = exchange.import_json("scooby.json")

    # Create Surface container
    surf_cont = multi.SurfaceContainer(surf_list)

    # Fix trim curves
    trimming.fix_multi_trim_curves(surf_cont)

    # Set sample size
    surf_cont.sample_size = 100

    # Set trim tessellator for all surfaces
    surf_cont.set_tessellator(tessellate.TrimTessellate)

    # Visualize
    surf_cont.vis = VisVTK.VisSurface(ctrlpts=False, legend=False, trims=False)
    surf_cont.render(evalcolor="blue", num_procs=16)

    # Export as .obj file
    # exchange.export_obj(surf_cont, "scooby.obj", update_delta=True)
