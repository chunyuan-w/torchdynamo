
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torchinductor.codecache import AsyncCompile

aten = torch.ops.aten
async_compile = AsyncCompile()


kernel0 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/i5/ci5zbqbzeij2usetynv7oczewshegubkvtpswwuumpp6xjync55y.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       const long ks0)
{
    #pragma omp parallel num_threads(56)
    {
        #pragma omp for 
        for(long i0=0; i0<344064*ks0; ++i0)
        {
            {
                {
                    auto tmp0 = in_ptr0[i0];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr0[i0] = tmp2;
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<344064; ++i0)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; ++i1)
            {
                {
                    {
                        auto tmp0 = out_ptr0[i1 + (i0*ks0)];
                        out_ptr1[i1 + (64*i0)] = tmp0;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<344064; ++i0)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; ++i1)
            {
                {
                    {
                        auto tmp0 = in_ptr1[i1 + (i0*ks0)];
                        out_ptr2[i1 + (64*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel1 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/i5/ci5zbqbzeij2usetynv7oczewshegubkvtpswwuumpp6xjync55y.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    #pragma omp parallel num_threads(56)
    {
        #pragma omp for 
        for(long i0=0; i0<344064; ++i0)
        {
            {
                {
                    float tmp3 = -std::numeric_limits<float>::infinity();
                    for(long i1=0; i1<384; ++i1)
                    {
                        {
                            auto tmp0 = in_ptr0[i1 + (384*i0)];
                            auto tmp1 = in_ptr1[i1 + (384*i0)];
                            auto tmp2 = tmp0 + tmp1;
                            tmp3 = std::max(tmp3, tmp2);
                        }
                    }
                    out_ptr0[i0] = tmp3;
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<344064; ++i0)
        {
            {
                {
                    float tmp6 = 0;
                    for(long i1=0; i1<384; ++i1)
                    {
                        {
                            auto tmp0 = in_ptr0[i1 + (384*i0)];
                            auto tmp1 = in_ptr1[i1 + (384*i0)];
                            auto tmp3 = out_ptr0[i0];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = std::exp(tmp4);
                            out_ptr1[i1 + (384*i0)] = tmp5;
                            tmp6 += tmp5;
                        }
                    }
                    out_ptr2[i0] = tmp6;
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<344064; ++i0)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<384; ++i1)
            {
                {
                    {
                        auto tmp0 = out_ptr1[i1 + (384*i0)];
                        auto tmp1 = out_ptr2[i0];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr3[i1 + (384*i0)] = tmp2;
                    }
                }
            }
        }
    }
}
''')

from torch.utils.cpp_extension import load_inline

wrapper = (
'''
#include <dlfcn.h>
#include <assert.h>            
at::Tensor call(at::Tensor arg0_1, at::Tensor arg1_1, at::Tensor arg2_1) {
    auto arg0_1_size = arg0_1.sizes();
    auto s0 = arg0_1_size[0];
    auto s1 = arg0_1_size[1];
    auto s2 = arg0_1_size[2];
    auto s3 = arg0_1_size[3];
    auto buf0 = at::empty_strided({56, 16, 384, s3}, {6144*s3, 384*s3, s3, 1}); 
    auto buf1 = at::empty_strided({896, 384, 64}, {24576, 64, 1}); 
    auto buf2 = at::empty_strided({896, 64, 384}, {24576, 1, 64}); 
    auto kernel0_lib = dlopen("/tmp/torchinductor_chunyuan/xz/cxzvex5kwp56drduhk2f6qtxrd5t5jqufxqwz6dpoukm4zxyfi6j.so", RTLD_NOW);
    assert(kernel0_lib != nullptr);
    void (*kernel0)(const float*,const float*,float*,float*,float*,const long);
    *(void **) (&kernel0) = dlsym(kernel0_lib, "kernel");
    kernel0((float*)(arg0_1.data_ptr()), (float*)(arg1_1.data_ptr()), (float*)(buf0.data_ptr()), (float*)(buf1.data_ptr()), (float*)(buf2.data_ptr()), s3);
    auto buf3 = at::empty_strided({896, 384, 384}, {147456, 384, 1}); 
    at::bmm_out(buf3, buf1, buf2);
    auto buf4 = at::empty_strided({56, 16, 384, 1}, {6144, 384, 1, 344064}); 
    auto buf5 = at::empty_strided({56, 16, 384, 384}, {2359296, 147456, 384, 1}); 
    auto buf6 = at::empty_strided({56, 16, 384, 1}, {6144, 384, 1, 344064}); 
    auto buf7 = at::empty_strided({56, 16, 384, 384}, {2359296, 147456, 384, 1}); 
    auto kernel1_lib = dlopen("/tmp/torchinductor_chunyuan/qo/cqoq2f5ra62ahgm5fvtmdumg4k3iypyloqo4w3xtrjfeb4jfxj3w.so", RTLD_NOW);
    assert(kernel1_lib != nullptr);
    void (*kernel1)(const float*,const float*,float*,float*,float*,float*);
    *(void **) (&kernel1) = dlsym(kernel1_lib, "kernel");
    kernel1((float*)(buf3.data_ptr()), (float*)(arg2_1.data_ptr()), (float*)(buf4.data_ptr()), (float*)(buf5.data_ptr()), (float*)(buf6.data_ptr()), (float*)(buf7.data_ptr()));
    return buf7; }''' )
module = load_inline(name='inline_extension', cpp_sources=[wrapper], functions=['call'], extra_cflags=['-DCPU_CAPABILITY_AVX2 -march=native -O3 -ffast-math -fno-finite-math-only -fopenmp'])
call = module.call


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import print_performance
    arg0_1 = rand_strided({56, 16, 384, 64}, {393216, 24576, 64, 1}, device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided({56, 16, 384, 64}, {393216, 24576, 64, 1}, device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided({56, 16, 384, 384}, {2359296, 147456, 384, 1}, device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))
