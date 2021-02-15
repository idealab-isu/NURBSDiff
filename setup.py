from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

setup(
    name='torch_nurbs_eval',
    ext_modules=[
        CppExtension(name='curve_eval_cpp',
            sources=['torch_nurbs_eval/csrc/curve_eval.cpp','torch_nurbs_eval/csrc/utils.cpp'],
            extra_include_paths=['torch_nurbs_eval/csrc/utils.h','torch_nurbs_eval/csrc/curve_eval.h']),
        CppExtension(name='surf_eval_cpp',
            sources=['torch_nurbs_eval/csrc/surf_eval.cpp','torch_nurbs_eval/csrc/utils.cpp'],
            extra_include_paths=['torch_nurbs_eval/csrc/utils.h','torch_nurbs_eval/csrc/surf_eval.h']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),)
