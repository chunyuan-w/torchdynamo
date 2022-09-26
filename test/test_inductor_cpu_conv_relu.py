import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_quantization import NodeSpec as ns, QuantizationTestCase
from torch.testing._internal.common_utils import run_tests

import torchdynamo        
from torchdynamo.testing import same
from torchinductor.compile_fx import compile_fx
from torchdynamo.optimizations.fuse_post_op import fuse_post_op, LinearEltwise

torchdynamo.config.raise_on_backend_error = False

def _clamp_modules():
    class MNoOpt(nn.Module):
        def __init__(self, m, in_channels, out_channels, bias, **kwargs):
            super(MNoOpt, self).__init__()
            self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

        def forward(self, x):
            x = self.conv(x)
            x = torch.clamp(x, min=-0.5, max=0.9)
            return x

    class MInf(nn.Module):
        def __init__(self, m, in_channels, out_channels, bias, **kwargs):
            super(MInf, self).__init__()
            self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

        def forward(self, x):
            x = self.conv(x)
            x = torch.clamp(x, min=0, max=float('inf'))
            return x

    class MNegInf(nn.Module):
        def __init__(self, m, in_channels, out_channels, bias, **kwargs):
            super(MNegInf, self).__init__()
            self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

        def forward(self, x):
            x = self.conv(x)
            x = torch.clamp(x, min=float('-inf'), max=0)
            return x

    class MOptMin(nn.Module):
        def __init__(self, m, in_channels, out_channels, bias, **kwargs):
            super(MOptMin, self).__init__()
            self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

        def forward(self, x):
            x = self.conv(x)
            x = torch.clamp(x, max=2)
            return x

    class MOptMax(nn.Module):
        def __init__(self, m, in_channels, out_channels, bias, **kwargs):
            super(MOptMax, self).__init__()
            self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

        def forward(self, x):
            x = self.conv(x)
            x = torch.clamp(x, min=0)
            return x

    return [MNoOpt, MInf, MNegInf, MOptMin, MOptMax]

def _eltwise_list():
    eltwise_list = [
        # [nn.ReLU(), 'aten::relu'],
        # [nn.Sigmoid(), 'aten::sigmoid'],
        # [nn.Tanh(), 'aten::tanh'],
        # [nn.Hardswish(), 'aten::hardswish'],
        # [nn.LeakyReLU(0.1, inplace=False), 'aten::leaky_relu'],
        # [nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False), 'aten::hardtanh'],
        # [nn.GELU(approximate="none"), 'aten::gelu'],
        # [nn.GELU(approximate="tanh"), 'aten::gelu'],

        # TODO: support inplace
        # [torch.relu, 'aten::relu'], # TODO support method relu
        # [F.relu, 'aten::relu'], # TODO support method relu
        [lambda x: x.relu(), 'aten::relu'],
        
        
        # [torch.sigmoid, 'aten::sigmoid'],
        # [torch.tanh, 'aten::tanh'],
    ]
    return eltwise_list


class TestFuseFx(QuantizationTestCase):

    def _test_conv_relu(self, eltwise_fn, bias, input_shape):
        class M(nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.linear(x)
                x = self.eltwise(x)
                return x    

        mod = M(eltwise_fn, input_shape[-1], 10, bias).eval()          

        @torchdynamo.optimize("inductor")
        def fn(x):
            return mod(x)

        v = torch.randn(input_shape)

        fused_gm = fuse_post_op(torch.fx.symbolic_trace(mod), [v])
        expected_nodes = [ns.call_module(LinearEltwise)]
        expected_occurrence = {
            # ns.call_module(nn.ReLU): 0
        }
        self.checkGraphModuleNodes(
            fused_gm,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)    

        # v = v.to(torch.bfloat16)
        # mod = mod.to(torch.bfloat16)

        print("#" * 50)
        result = fn(v)
        print("#" * 50)
        assert same(result, mod(v))

        print("run")
        fn(v)
        fn(v)


    def test(self):
        for eltwise_fn in _eltwise_list():
            for input_shape in [
                [2, 3, 10],
                [2, 10]]:
                for bias in [
                    True,
                    False]:
                    self._test_conv_relu(eltwise_fn[0], bias, input_shape)
        print("done")

# # clamp:
# modules = _clamp_modules()
# for M in modules:
#     for input_shape in [
#         [2, 3, 10],
#         [2, 10]]:
#         for bias in [
#             True,
#             False]:
#             mod = M(nn.Linear, input_shape[-1], 10, bias)

#             mod.eval()

#             # TODO: fix duplicate
#             @torchdynamo.optimize("inductor")
#             def fn(x):
#                 return mod(x)

#             v = torch.randn(input_shape)

#             # v = v.to(torch.bfloat16)
#             # mod = mod.to(torch.bfloat16)

#             print("#" * 50)
#             result = fn(v)
#             print("#" * 50)
#             assert same(result, mod(v))

#             print("run")
#             fn(v)
#             fn(v)            


if __name__ == '__main__':
    run_tests()