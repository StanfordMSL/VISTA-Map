# setup.py
from setuptools import setup, find_packages

setup(
    name="vista",
    version="0.1.0",
    author="Timothy Chen",
    author_email="chengine@stanford.edu",
    description="GPU-based voxel ray traversal in Pytorch",
    packages=find_packages(),
    install_requires=[
        'open3d == 0.18.0',
        'tensordict == 0.3.0',
        'matplotlib'
    ]
)