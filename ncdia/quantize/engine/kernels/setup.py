from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="quant_engine",
    ext_modules=[
        CUDAExtension(
            name="quant_engine",
            sources=[
                "pybind.cpp",
                "tpack/tpack.cu",
                "functions/linear.cu",
                "functions/quantlinear.cu",
            ]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
