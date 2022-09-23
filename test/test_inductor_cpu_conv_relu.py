import torch
import torchdynamo        
from torchdynamo.testing import same

torchdynamo.config.raise_on_backend_error = False

def test_conv_relu(bias, input_shape):
    mod = torch.nn.Sequential(
        torch.nn.Linear(input_shape[-1], 10, bias=bias),
        torch.nn.ReLU(),
    ).eval()            

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


for input_shape in [
    [2, 3, 10],
    [2, 10]]:
    for bias in [
        True,
        False]:
        test_conv_relu(bias, input_shape)

print("done")