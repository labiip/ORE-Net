from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='areconv',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='areconv.ext',
            sources=[
                'areconv/extensions/extra/cloud/cloud.cpp',
                'areconv/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'areconv/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'areconv/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'areconv/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'areconv/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
