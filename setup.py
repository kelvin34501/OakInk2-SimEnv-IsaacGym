from setuptools import setup, find_packages
from distutils.extension import Extension

setup(
    name="oakink2-simenv-isaacgym",
    version="0.0.1",
    python_requires=">=3.8.0",
    packages=find_packages(
        where="src",
        include="dyn_mf*,dev_util*",
    ),
    package_dir={"": "src"},
    include_package_data=True,
)
