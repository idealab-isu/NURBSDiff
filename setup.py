import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension,CUDAExtension

try:
    setup(
        name='NURBSDiff',
        ext_modules=[
            CppExtension(name='NURBSDiff.curve_eval_cpp',
                sources=['NURBSDiff/csrc/curve_eval.cpp','NURBSDiff/csrc/utils.cpp'],
                extra_include_paths=['NURBSDiff/csrc/utils.h','NURBSDiff/csrc/curve_eval.h']),
            CppExtension(name='NURBSDiff.surf_eval_cpp',
                sources=['NURBSDiff/csrc/surf_eval.cpp','NURBSDiff/csrc/utils.cpp'],
                extra_include_paths=['NURBSDiff/csrc/utils.h','NURBSDiff/csrc/surf_eval.h']),
            CUDAExtension(name='NURBSDiff.curve_eval_cuda',
                sources=['NURBSDiff/csrc/curve_eval_cuda.cpp',
                'NURBSDiff/csrc/curve_eval_cuda_kernel.cu']),
            CUDAExtension(name='NURBSDiff.surf_eval_cuda',
                sources=['NURBSDiff/csrc/surf_eval_cuda.cpp',
                'NURBSDiff/csrc/surf_eval_cuda_kernel.cu']),
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        packages=find_packages(),)
except:
    print('installation of NURBSDiff with GPU wasnt successful, installing CPU version')
    setup(
        name='NURBSDiff',
        ext_modules=[
            CppExtension(name='NURBSDiff.curve_eval_cpp',
                sources=['NURBSDiff/csrc/curve_eval.cpp','NURBSDiff/csrc/utils.cpp'],
                extra_include_paths=['NURBSDiff/csrc/utils.h','NURBSDiff/csrc/curve_eval.h']),
            CppExtension(name='NURBSDiff.surf_eval_cpp',
                sources=['NURBSDiff/csrc/surf_eval.cpp','NURBSDiff/csrc/utils.cpp'],
                extra_include_paths=['NURBSDiff/csrc/utils.h','NURBSDiff/csrc/surf_eval.h']),
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        packages=find_packages(),)