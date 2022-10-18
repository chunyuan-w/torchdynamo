import math
import torch
import torch.nn as nn
import numpy as np

# import torch._dynamo
import torchdynamo
from torch._inductor import config

config.cpp.simdlen = 8
config.dynamic_shapes = True


@torchdynamo.optimize()
def test_softmax(a):
	return torch.nn.functional.softmax(a)
	# return torch.matmul(a, torch.transpose(a, 0, 1))


is_contiguous = True
shape = (2, 512)
a = torch.rand(shape)
n_iter = 5

for _ in range(n_iter):
	test_softmax(a)


dynamo_result = test_softmax(a)
ref_result = torch.nn.functional.softmax(a)
print(dynamo_result)
print(ref_result)

np.testing.assert_allclose(dynamo_result.numpy(), ref_result.numpy(), atol=1e-6)

# class MHAScoresCalculation(nn.Module):
# 	def __init__(self, dim_per_head, softmax_dim=-1):
# 			super(MHAScoresCalculation, self).__init__()
# 			self.softmax = nn.Softmax(dim=softmax_dim)
# 			self.dim_per_head = dim_per_head
	
# 	def forward(self, mat1, mat2, bias):
# 		mat1 = mat1 / math.sqrt(self.dim_per_head)
# 		qk = torch.matmul(mat1, mat2.transpose(2, 3))
# 		scores = qk + bias
# 		return self.softmax(scores)



# seq_length = 384
# mat1 = torch.randn(56, 16, seq_length, 64)
# mat2 = torch.randn(56, 16, seq_length, 64)
# bias = torch.randn(56, 16, seq_length, seq_length)
# with torch.no_grad():
# 	mha = MHAScoresCalculation(64)
# 	optimized_mod = torchdynamo.optimize()(mha)
# 	res_ref = optimized_mod(mat1, mat2, bias)



print("#" * 50)
