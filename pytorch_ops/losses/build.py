import torch
import os
from torch.utils.ffi import create_extension
this_file = os.path.dirname(os.path.realpath(__file__))

extra_objects = ['cd/cd_cuda_kernel.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
ffi = create_extension(
    name='_ext.cd',
    headers=['cd/cd_cuda.h'],
    sources=['cd/cd_cuda.c'],
    define_macros=[('WITH_CUDA', None)],
    with_cuda=True,
    relative_to=__file__,
    extra_objects=extra_objects,
    extra_compile_args=["-I/usr/local/cuda-8.0/include"])
ffi.build()

extra_objects = ['emd/emd_cuda_kernel.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
ffi1 = create_extension(
    name='_ext.emd',
    headers=['emd/emd_cuda.h'],
    sources=['emd/emd_cuda.c'],
    define_macros=[('WITH_CUDA', None)],
    with_cuda=True,
    relative_to=__file__,
    extra_objects=extra_objects,
    extra_compile_args=["-I/usr/local/cuda-8.0/include"])
ffi1.build()

