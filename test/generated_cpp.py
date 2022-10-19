
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torchinductor.codecache import AsyncCompile

aten = torch.ops.aten
async_compile = AsyncCompile()


kernel0 = ('''
#include "/tmp/torchinductor_chunyuan/i5/ci5zbqbzeij2usetynv7oczewshegubkvtpswwuumpp6xjync55y.h"
extern "C" void kernel0(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       const long ks0,
                       const long ks1)
{
    #pragma GCC ivdep
    for(long i0=0; i0<ks0; ++i0)
    {
        {
            {
                float tmp1 = -std::numeric_limits<float>::infinity();
                for(long i1=0; i1<ks1; ++i1)
                {
                    {
                        auto tmp0 = in_ptr0[i1 + (i0*ks1)];
                        tmp1 = std::max(tmp1, tmp0);
                    }
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    #pragma GCC ivdep
    for(long i0=0; i0<ks0; ++i0)
    {
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<ks1; ++i1)
                {
                    {
                        auto tmp0 = in_ptr0[i1 + (i0*ks1)];
                        auto tmp1 = out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = std::exp(tmp2);
                        out_ptr1[i1 + (i0*ks1)] = tmp3;
                        tmp4 += tmp3;
                    }
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    #pragma GCC ivdep
    for(long i0=0; i0<ks0; ++i0)
    {
        #pragma GCC ivdep
        for(long i1=0; i1<ks1; ++i1)
        {
            {
                {
                    auto tmp0 = out_ptr1[i1 + (i0*ks1)];
                    auto tmp1 = out_ptr2[i0];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr3[i1 + (i0*ks1)] = tmp2;
                }
            }
        }
    }
}
''')

from torch.utils.cpp_extension import load_inline

wrapper = (
'''
at::Tensor call(at::Tensor arg0_1) {
    auto arg0_1_size = arg0_1.sizes();
    auto s0 = arg0_1_size[0];
    auto s1 = arg0_1_size[1];
    auto buf0 = at::empty_strided({s0, 1}, {1, s0}); 
    auto buf1 = at::empty_strided({s0, s1}, {s1, 1}); 
    auto buf2 = at::empty_strided({s0, 1}, {1, s0}); 
    auto buf3 = at::empty_strided({s0, s1}, {s1, 1}); 
    kernel0((float*)(arg0_1.data_ptr()), (float*)(buf0.data_ptr()), (float*)(buf1.data_ptr()), (float*)(buf2.data_ptr()), (float*)(buf3.data_ptr()), s0, s1);
    return buf3; }''' )
module = load_inline(name='inline_extension', cpp_sources=[kernel0, wrapper], functions=['call'], extra_cflags=['-DCPU_CAPABILITY_AVX2 -march=native -O3 -ffast-math -fno-finite-math-only -fopenmp'])
call = module.call


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import print_performance
    arg0_1 = rand_strided({2, 512}, {512, 1}, device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))
