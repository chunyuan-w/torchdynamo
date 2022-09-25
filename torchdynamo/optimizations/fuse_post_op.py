import copy

import torch
import torch.fx as fx
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module


class LinearEltwiseOneOperand(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(LinearEltwiseOneOperand, self).__init__(in_features, out_features, bias=bias,
            device=device, dtype=dtype)
    
    def forward(self, input):
        y = torch.ops.mkldnn_prepacked.linear_relu(input, self.weight, self.bias, self.attr, self.scalars, self.algorithm)
        return y

def fuse_linear_eltwise_eval(linear, eltwise, attr):
    linear_relu = LinearEltwiseOneOperand(linear.in_features,
                              linear.out_features,
                              linear.bias is not None,
                              linear.weight.device,
                              linear.weight.dtype)
    linear_relu.__dict__ = copy.deepcopy(linear.__dict__)
    # TODO: set this in init func is not working, due to copy __dict__??
    linear_relu.attr = attr
    linear_relu.scalars = []
    linear_relu.algorithm = ""
    # TODO: define this behavior with a dict?
    if attr == "leaky_relu":
        linear_relu.scalars = [eltwise.negative_slope]
    if attr == "hardtanh":
        linear_relu.scalars = [eltwise.min_val, eltwise.max_val]
    if attr == "gelu":
        linear_relu.algorithm = eltwise.approximate
    return linear_relu

def fuse_post_op(gm, example_inputs):
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    patterns = [
        (torch.nn.Linear, torch.nn.ReLU),
        (torch.nn.Linear, torch.nn.Sigmoid),
        (torch.nn.Linear, torch.nn.Tanh),
        (torch.nn.Linear, torch.nn.Hardswish),
        (torch.nn.Linear, torch.nn.LeakyReLU),
        (torch.nn.Linear, torch.nn.Hardtanh),
        (torch.nn.Linear, torch.nn.GELU),
    ]
    attr_names = [
        "relu",
        "sigmoid",
        "tanh",
        "hardswish",
        "leaky_relu",
        "hardtanh",
        "gelu",
    ]
    assert len(patterns) == len(attr_names), "pattern and replacement length should be equal"
    for pattern, attr_name in zip(patterns, attr_names):
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of bn is used by other nodes
                    continue
                linear = modules[node.args[0].target]
                eltwise = modules[node.target]
                eval_mode = all(not n.training for n in [linear, eltwise])

                tensors = example_inputs + [linear.weight]
                if linear.bias is not None:
                    tensors.append(linear.bias)
                is_cpu = all(x.device == torch.device('cpu') for x in tensors)
                if eval_mode and is_cpu:
                    fused_linear = fuse_linear_eltwise_eval(linear, eltwise, attr_name)
                    replace_node_module(node.args[0], modules, fused_linear)
                    node.replace_all_uses_with(node.args[0])
                    new_graph.erase_node(node)
    gm =  fx.GraphModule(gm, new_graph)    
    return gm