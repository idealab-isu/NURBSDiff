from setuptools import setup, find_packages

setup(
    name='NURBSDiff',
    version='2.0.0',
    description='Pure PyTorch implementation for differentiable NURBS curve and surface evaluation',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-repo/NURBSDiff',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'einops>=0.6.0',
    ],
    extras_require={
        'examples': ['pytorch3d', 'geomdl', 'matplotlib', 'tqdm'],
        'dev': ['pytest>=7.0.0', 'pytest-cov'],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)