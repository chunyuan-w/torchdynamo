import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_utils import run_tests

import torchdynamo
from torchdynamo.testing import same
from torchinductor.overrides import LinearEltwise
from torchinductor.overrides import fuse_fx

torchdynamo.config.raise_on_backend_error = False


def _eltwise_list():
    eltwise_list = [
        nn.ReLU(),
        torch.relu,
        F.relu,
        lambda x: x.relu(),
        nn.Sigmoid(),
        torch.sigmoid,
        F.sigmoid,
        lambda x: x.sigmoid(),
        nn.Tanh(),
        torch.tanh,
        F.tanh,
        lambda x: x.tanh(),
        nn.Hardswish(),
        F.hardswish,
        nn.LeakyReLU(0.1, inplace=False),
        lambda x: F.leaky_relu(x, negative_slope=0.2),
        # lambda x: F.leaky_relu(x, 0.2), # TODO: only works for kwargs but not args
        F.leaky_relu,
        nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False),
        # lambda x: F.hardtanh(x, -2.0, max_val=5), # TODO: not work with mixed args and kwargs
        F.hardtanh,
        nn.GELU(approximate="none"),
        nn.GELU(approximate="tanh"),
        lambda x: F.gelu(x, approximate="none"),
        lambda x: F.gelu(x, approximate="tanh"),
        F.gelu,
    ]
    return eltwise_list


# Inherit the QuantizationTestCase class
# to leverage the checkGraphModuleNodes function
class TestFuseFx(QuantizationTestCase):
    def _test_linear_eltwise(self, eltwise_fn, bias, input_shape):
        class M(nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
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

        fused_gm = fuse_fx(torch.fx.symbolic_trace(mod), [v])
        expected_nodes = [ns.call_module(LinearEltwise)]
        self.checkGraphModuleNodes(fused_gm, expected_node_list=expected_nodes)

        result = fn(v)
        assert same(result, mod(v))

    def test_linear_eltwise(self):
        for eltwise_fn in _eltwise_list():
            for input_shape in [[2, 3, 10], [2, 10]]:
                for bias in [True, False]:
                    self._test_linear_eltwise(eltwise_fn, bias, input_shape)


if __name__ == "__main__":
    run_tests()
