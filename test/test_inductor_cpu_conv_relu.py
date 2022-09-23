import torch
import torchdynamo        
from torchdynamo.testing import same
        
def test_conv_relu():
    mod = torch.nn.Sequential(
        torch.nn.Conv2d(10, 10, 5),
        torch.nn.ReLU(),
    ).eval()            

    @torchdynamo.optimize("inductor")
    def fn(x):
        return mod(x)

    v = torch.randn(1, 10, 14, 14)
    print("#" * 50)
    result = fn(v)
    print("#" * 50)
    assert same(result, mod(v))

test_conv_relu()