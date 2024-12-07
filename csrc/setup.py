from setuptools import setup
from torch.utils import cpp_extension
import glob
import os

source_files = glob.glob(os.path.join('vllm_csrc/attention', '**', '*.cu'), recursive=True) + \
               glob.glob(os.path.join('vllm_csrc/attention', '**', '*.cpp'), recursive=True)
source_files_filtered = []

for s in source_files:
    if "torch_bindings" not in s:# and "expose" not in s:
        source_files_filtered.append(s)

pos_encoding_files = ["vllm_csrc/pos_encoding_kernels_k.cu",
                      "vllm_csrc/pos_encoding_kernels_k_fused.cu",
                      "vllm_csrc/pos_encoding_kernels_k_fused_paged.cu"]         

setup(
    name='lmc_ops',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'lmc_ops',
            [
                'pybind.cpp',
                'mem_kernels.cu',
                'cal_cdf.cu',
                'ac_enc.cu',
                'ac_dec.cu',
            ] + source_files_filtered + \
            pos_encoding_files,
            #extra_compile_args={'cxx': ['-g'],
            #                    'nvcc': ['-G', '-g']},
            #include_dirs=['./']#['./include']
        ),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
