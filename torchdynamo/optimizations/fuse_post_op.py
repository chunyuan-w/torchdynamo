import copy

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module


class LinearEltwise(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(LinearEltwise, self).__init__(in_features, out_features, bias=bias,
            device=device, dtype=dtype)
    
    def forward(self, input):
        y = torch.ops.mkldnn_prepacked.linear_eltwise(input, self.weight, self.bias, self.attr, self.scalars, self.algorithm)
        return y

    def update_status(self, eltwise, attr, extra_inputs):
        self.attr = attr

        scalars = []
        for item in extra_inputs.scalars:
            assert hasattr(eltwise, item)
            scalars.append(getattr(eltwise, item))
        self.scalars = scalars

        algorithm = ""
        if extra_inputs.algorithm:
            assert hasattr(eltwise, extra_inputs.algorithm)
            algorithm = getattr(eltwise, extra_inputs.algorithm) 
        self.algorithm = algorithm

def fuse_linear_eltwise_eval(linear, eltwise, attr, extra_inputs):
    linear_eltwise = LinearEltwise(linear.in_features,
                              linear.out_features,
                              linear.bias is not None,
                              linear.weight.device,
                              linear.weight.dtype)
    linear_eltwise.__dict__ = copy.deepcopy(linear.__dict__)
    # TODO: set this in init func is not working, due to copy __dict__??
    linear_eltwise.update_status(eltwise, attr, extra_inputs)
    return linear_eltwise

class EltwiseFusionOp:
    def __init__(self, post_op, scalars=[], algorithm=""):
        self.post_op = post_op
        self.scalars = scalars
        self.algorithm = algorithm

computation_op = nn.Linear

attr_map = {
    "relu": EltwiseFusionOp(post_op=nn.ReLU()),
    "sigmoid": EltwiseFusionOp(post_op=nn.Sigmoid()),
    "tanh": EltwiseFusionOp(post_op=nn.Tanh),
    "hardswish": EltwiseFusionOp(post_op=nn.Hardswish),
    "leaky_relu": EltwiseFusionOp(post_op=nn.LeakyReLU, scalars=["negative_slope"]),
    "hardtanh": EltwiseFusionOp(post_op=nn.Hardtanh, scalars=["min_val", "max_val"]),
    "gelu": EltwiseFusionOp(post_op=nn.GELU, algorithm="approximate"),
}

def fuse_post_op(gm, example_inputs):
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for attr_name, value in attr_map.items():
        pattern = (computation_op, value.post_op)
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of linear is used by other nodes
                    continue
                linear = modules[node.args[0].target]
                eltwise = modules[node.target]
                eval_mode = all(not n.training for n in [linear, eltwise])

                tensors = example_inputs + [linear.weight]
                if linear.bias is not None:
                    tensors.append(linear.bias)
                is_cpu = all(x.device == torch.device('cpu') for x in tensors)
                if eval_mode and is_cpu:
                    fused_linear = fuse_linear_eltwise_eval(linear, eltwise, attr_name, value)
                    replace_node_module(node.args[0], modules, fused_linear)
                    node.replace_all_uses_with(node.args[0])
                    new_graph.erase_node(node)
    gm =  fx.GraphModule(gm, new_graph)    
    return gm