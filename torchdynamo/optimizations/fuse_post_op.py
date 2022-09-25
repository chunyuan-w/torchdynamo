import copy

import torch
import torch.fx as fx
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module


class LinearReLU(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(LinearReLU, self).__init__(in_features, out_features, bias=bias,
            device=device, dtype=dtype)
    
    def forward(self, input):
        y = torch.ops.mkldnn_prepacked.linear_relu(input, self.weight, self.bias, "relu")
        return y

def fuse_linear_relu_eval(linear, relu):
    linear_relu = LinearReLU(linear.in_features,
                              linear.out_features,
                              linear.bias is not None,
                              linear.weight.device,
                              linear.weight.dtype)
    linear_relu.__dict__ = copy.deepcopy(linear.__dict__)
    return linear_relu

def fuse_post_op(gm, example_inputs):
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    patterns = [(torch.nn.Linear, torch.nn.ReLU)]
    replacements = [fuse_linear_relu_eval]
    assert len(patterns) == len(replacements), "pattern and replacement length should be equal"
    for pattern, replace_func in zip(patterns, replacements):
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of bn is used by other nodes
                    continue
                linear = modules[node.args[0].target]
                relu = modules[node.target]
                eval_mode = all(not n.training for n in [linear, relu])

                tensors = example_inputs + [linear.weight]
                if linear.bias is not None:
                    tensors.append(linear.bias)
                is_cpu = all(x.device == torch.device('cpu') for x in tensors)
                if eval_mode and is_cpu:
                    fused_linear = replace_func(linear, relu)
                    replace_node_module(node.args[0], modules, fused_linear)
                    node.replace_all_uses_with(node.args[0])
                    new_graph.erase_node(node)
    gm =  fx.GraphModule(gm, new_graph)    
    return gm