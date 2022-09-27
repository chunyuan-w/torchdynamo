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


def _supported_eltwise_list():
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
        nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False),
        nn.GELU(approximate="none"),
        nn.GELU(approximate="tanh"),
        F.gelu,
        lambda x: F.gelu(x, approximate="none"),
        lambda x: F.gelu(x, approximate="tanh"),
    ]
    return eltwise_list


def _unsupported_eltwise_list():
    eltwise_list = [
        F.leaky_relu,
        F.hardtanh,
        lambda x: F.hardtanh(x, 2, max_val=3),
    ]
    return eltwise_list


# Inherit the QuantizationTestCase class
# to leverage the checkGraphModuleNodes function
class TestFuseFx(QuantizationTestCase):
    def _test_linear_eltwise(self, eltwise_fn, bias, input_shape, expected_node):
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
        expected_nodes = [ns.call_module(expected_node)]
        self.checkGraphModuleNodes(fused_gm, expected_node_list=expected_nodes)

        result = fn(v)
        assert same(result, mod(v))

    def test_linear_eltwise(self):
        for eltwise_fn in _supported_eltwise_list():
            for input_shape in [[2, 3, 10], [2, 10]]:
                for bias in [True, False]:
                    self._test_linear_eltwise(
                        eltwise_fn, bias, input_shape, expected_node=LinearEltwise
                    )

    def test_unsupported_linear_eltwise(self):
        for eltwise_fn in _unsupported_eltwise_list():
            self._test_linear_eltwise(
                eltwise_fn, bias=True, input_shape=[2, 3, 10], expected_node=nn.Linear
            )


if __name__ == "__main__":
    run_tests()
