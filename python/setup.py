from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
import os
import glob

# CUDA setup
def find_cuda():
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Try common locations
        for path in ['/usr/local/cuda', '/opt/cuda', '/usr/lib/cuda']:
            if os.path.exists(path):
                cuda_home = path
                break
    
    if cuda_home is None:
        raise RuntimeError("CUDA installation not found. Please set CUDA_HOME environment variable.")
    
    return cuda_home

cuda_home = find_cuda()
cuda_include = os.path.join(cuda_home, 'include')
cuda_lib = os.path.join(cuda_home, 'lib64')

# Determine compute capability
def get_compute_capability():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            compute_cap = result.stdout.strip().split('\n')[0]
            return compute_cap.replace('.', '')
    except:
        pass
    
    # Default to compute capability 7.5 (RTX 20xx series)
    return '75'

compute_cap = get_compute_capability()

# Source files
cpp_sources = ['fpn_pytorch.cpp']
cuda_sources = [
    '../src/kernels/unified_fpn_kernels.cu',
    '../src/core/fpn_core.cu'
]

# Include directories
include_dirs = [
    pybind11.get_include(),
    torch.utils.cpp_extension.include_paths()[0],
    '../include',
    cuda_include,
    '/usr/local/include'  # For system libraries like CUB
]

# Library directories
library_dirs = [
    torch.utils.cpp_extension.library_paths()[0],
    cuda_lib
]

# Libraries
libraries = [
    'torch',
    'torch_python', 
    'cudart',
    'cublas',
    'cudnn',
    'curand'
]

# Compiler flags
extra_compile_args = {
    'cxx': [
        '-O3',
        '-std=c++17',
        '-fPIC',
        '-DWITH_CUDA',
        '-DTORCH_EXTENSION_NAME=unified_fpn'
    ],
    'nvcc': [
        '-O3',
        f'-gencode=arch=compute_{compute_cap},code=sm_{compute_cap}',
        '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
        '-gencode=arch=compute_80,code=sm_80',  # A100
        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-use_fast_math',
        '-Xptxas=-v',
        '-Xcompiler=-fPIC',
        '-std=c++17',
        '-DWITH_CUDA',
        '-DTORCH_EXTENSION_NAME=unified_fpn'
    ]
}

# Linker flags  
extra_link_args = [
    '-L' + cuda_lib,
    '-lcudart',
    '-lcublas', 
    '-lcudnn',
    '-lcurand'
]

# Custom build extension to handle CUDA
class CUDAExtension(Pybind11Extension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_extension(self, ext):
        # Set CUDA compiler
        os.environ['CC'] = 'gcc'
        os.environ['CXX'] = 'g++'
        os.environ['NVCC'] = os.path.join(cuda_home, 'bin', 'nvcc')
        
        super().build_extension(ext)

# Main extension definition
ext_modules = [
    CUDAExtension(
        'unified_fpn',
        sources=cpp_sources + cuda_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
]

setup(
    name='unified_fpn',
    version='1.0.0',
    author='Advanced CUDA Developer',
    author_email='dev@example.com',
    description='High-performance unified CUDA kernel for Feature Pyramid Networks',
    long_description='''
    A highly optimized CUDA implementation of Feature Pyramid Networks (FPN) that fuses
    multiple operations into single kernels for maximum performance. Designed for real-time
    object detection and instance segmentation applications.
    
    Features:
    - Fused lateral convolution + upsampling + element-wise addition
    - Memory-optimized shared memory usage
    - Multi-stream parallel execution
    - Support for FP32/FP16 precision
    - PyTorch integration with automatic differentiation
    - Comprehensive benchmarking and profiling tools
    ''',
    long_description_content_type='text/plain',
    url='https://github.com/username/unified-fpn',
    packages=['unified_fpn'],
    package_dir={'unified_fpn': '.'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.20.0',
        'pybind11>=2.6.0'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'isort>=5.0',
            'mypy>=0.900'
        ],
        'benchmark': [
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'pandas>=1.3.0'
        ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='cuda pytorch fpn object-detection computer-vision gpu-acceleration',
    project_urls={
        'Bug Reports': 'https://github.com/username/unified-fpn/issues',
        'Source': 'https://github.com/username/unified-fpn',
        'Documentation': 'https://unified-fpn.readthedocs.io/'
    },
    zip_safe=False
)