from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
import numpy as np
import glob

# torch ext
torch_ext = CUDAExtension(
    name="dev_fn.geo.ops._C",
    sources=glob.glob("src/dev_fn/csrc/**/*.cpp", recursive=True)
    + glob.glob("src/dev_fn/csrc/**/*.cu", recursive=True),
    define_macros=[
        ("WITH_CUDA", None),
        ("THRUST_IGNORE_CUB_VERSION_CHECK", None),
    ],
    extra_compile_args={
        "nvcc": [
            "-O3",
            "-DWITH_CUDA",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        ]
    },
    include_dirs=[],
)

# get np include path
np_include_path = np.get_include()

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    "dev_fn.geo.ops.triangle_hash",
    sources=["src/dev_fn/cython/triangle_hash.pyx"],
    libraries=["m"],  # Unix-like specific
    include_dirs=[np_include_path],
)

# Gather all extension modules
cython_ext_modules = [
    triangle_hash_module,
]
cython_ext_modules = cythonize(cython_ext_modules)
torch_ext_modules = [torch_ext]
ext_modules = torch_ext_modules + cython_ext_modules

setup(
    name="oakink2-simenv-isaacgym",
    version="0.0.1",
    python_requires=">=3.8.0",
    packages=find_packages(
        where="src",
        include="dyn_mf*,dev_fn*",
    ),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
