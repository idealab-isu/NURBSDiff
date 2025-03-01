# NURBSDiff
This repo contains code for fitting curves and surfaces to any input point cloud.

# Requirements and Install dependencies

## Dependencies
1. Pytorch: Installation command can be generated from [here](https://pytorch.org/get-started/locally/).
2. Pytorch 3D:
	* For CPU only install `pip install pytorch3d` should do
	* For macOS running on Apple Silicon `MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git"`
	* For GPU support, we would need to install `pytorch3d` using the following process
				```
				TBD
				pip install "git+https://github.com/facebookresearch/pytorch3d.git"
				```

* Geomdl: `pip install geomdl`

<!-- # Installation of the package
The following commands need to be modified to compile the code successfully, with pytorch code as well.

```
sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/api/module.h
sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/runtime/argument_spec.h
sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/pybind11/cast.h
```

* Now proceed to run TorchNURBSEval by the following command:
`call "%VS2017INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 10.0.17763.0 && set DISTUTILS_USE_SDK=1 && set PY_VCRUNTIME_REDIST=No thanks && set MSSdk=1 && python setup.py develop`
or open x64 Native Tools Command Prompt for VS2017 and run the following command from the TorchNURBSEval folder.
`python setup.py develop`
 -->
 
# Usage of NURBSDiff 

* Curve Evaluation (curve_eval.py)
  1. The evaluation kernels for curve_eval.py are written under torch_nurbs_eval/csrc/curve_eval.cpp
  2. To run curve_eval.py, provide input control points, input point cloud and set the number of evaluation points under out_dim in CurveEval.
	3. To generate random distribution of control points, use data_generator.gen_control_points()
	4. Input Size parameters:
	    * control points : (No of curves, no of control points, [(x,y,weights) or (x,y,z,weights)] )
	    * point cloud : (No of point clouds, no of points in point cloud,3)
	    * Parameters to vary: degree, number of control points, number of evaluation points.
	5. To run the curve evaluation, cd into torch_nurbs_eval.
	6. To run `python curve_eval.py`

(Will add details for Surface Fitting soon)
