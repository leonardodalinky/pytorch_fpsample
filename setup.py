import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension

__version__ = "0.1.0"
# WITH_CUDA = os.getenv("WITH_CUDA", "1") == "1"
WITH_CUDA = False

sources = [
    "csrc/fpsample.cpp",
    "csrc/cpu/fpsample_cpu.cpp",
    "csrc/cpu/bucket_fps/wrapper.cpp",
]


if not WITH_CUDA:
    ext_modules = [
        cpp_extension.CppExtension(
            name="torch_fpsample._core",
            include_dirs=["csrc"],
            sources=sources,
        )
    ]
else:
    # TODO
    sources += []
    ext_modules = [
        cpp_extension.CUDAExtension(
            name="torch_fpsample._core",
            include_dirs=["csrc"],
            sources=sources,
        )
    ]


setup(
    name="torch_fpsample",
    version=__version__,
    author="Leonard Lin",
    author_email="leonard.keilin@gmail.com",
    description="PyTorch implementation of fpsample.",
    ext_modules=ext_modules,
    keywords=["pytorch", "farthest", "furthest", "sampling", "sample", "fps"],
    packages=find_packages(),
    package_data={"": ["*.pyi"]},
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
