import torch
import torch.nn as nn
import torchdynamo        
from torchdynamo.testing import same

torchdynamo.config.raise_on_backend_error = False

def _eltwise_list():
    eltwise_list = [
        [nn.ReLU(), 'aten::relu'],
        [nn.Sigmoid(), 'aten::sigmoid'],

        # [torch.relu, 'aten::relu'], # TODO support method relu
        # [torch.sigmoid, 'aten::sigmoid'],
        # [torch.tanh, 'aten::tanh'],
        # [nn.LeakyReLU(0.1, inplace=False), 'aten::leaky_relu'],
        # [nn.Hardtanh(inplace=False), 'aten::hardtanh'],
        # [nn.GELU(approximate="none"), 'aten::gelu'],
        # [nn.GELU(approximate="tanh"), 'aten::gelu'],
    ]
    return eltwise_list

def test_conv_relu(eltwise_fn, bias, input_shape):
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

    # v = v.to(torch.bfloat16)
    # mod = mod.to(torch.bfloat16)

    print("#" * 50)
    result = fn(v)
    print("#" * 50)
    assert same(result, mod(v))

    print("run")
    fn(v)
    fn(v)

for eltwise_fn in _eltwise_list():
    for input_shape in [
        [2, 3, 10],
        [2, 10]]:
        for bias in [
            True,
            False]:
            test_conv_relu(eltwise_fn[0], bias, input_shape)

print("done")