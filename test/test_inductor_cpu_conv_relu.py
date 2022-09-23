import torch
import torchdynamo        
from torchdynamo.testing import same

torchdynamo.config.raise_on_backend_error = False

def test_conv_relu():
    mod = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
    ).eval()            

    @torchdynamo.optimize("inductor")
    def fn(x):
        return mod(x)

    v = torch.randn(2, 10)
    print("#" * 50)
    result = fn(v)
    print("#" * 50)
    assert same(result, mod(v))

test_conv_relu()