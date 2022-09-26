import copy

import torch
import torch.nn.functional as F

import torch.fx as fx
from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module
from typing import Type, Dict, Any, Tuple, Iterable


class LinearEltwise(torch.nn.Linear):
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

def matches_module_method_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    if len(pattern) != len(nodes):
        return False
    if len(pattern) != 2:
        return False
    
    node_linear = nodes[0]
    node_relu = nodes[1]

    pattern_linear = pattern[0]
    pattern_relu = pattern[1]

    if not isinstance(node_linear, fx.Node) or not isinstance(node_relu, fx.Node):
        return False
    if node_linear.op != 'call_module':
        return False
    if node_relu.op != 'call_function' and node_relu.op != 'call_method':
        return False
    if not isinstance(node_linear.target, str):
        return False
    if node_linear.target not in modules:
        return False
    if type(modules[node_linear.target]) is not pattern_linear:
        return False
    
    if node_relu.target != pattern_relu:
        return False
    return True

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
    def __init__(self, scalars=[], algorithm=""):
        self.scalars = scalars
        self.algorithm = algorithm

attr_names = {
    "relu": EltwiseFusionOp(),
    # "sigmoid": EltwiseFusionOp(),
    # "tanh": EltwiseFusionOp(),
    # "hardswish": EltwiseFusionOp(),
    # "leaky_relu": EltwiseFusionOp(scalars=["negative_slope"]),
    # "hardtanh": EltwiseFusionOp(scalars=["min_val", "max_val"]),
    # "gelu": EltwiseFusionOp(algorithm="approximate"),
}

def fuse_post_op(gm, example_inputs):
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    patterns = [
        # (torch.nn.Linear, torch.relu),
        # (torch.nn.Linear, F.relu),
        (torch.nn.Linear, 'relu'),
        # (torch.nn.Linear, torch.nn.ReLU()),

        # TODO: if not a module but a function, how to pass the func call values to the new module: like leakyrelu
        # (torch.nn.Linear, torch.nn.Sigmoid),
        # (torch.nn.Linear, torch.nn.Tanh),
        # (torch.nn.Linear, torch.nn.Hardswish),
        # (torch.nn.Linear, torch.nn.LeakyReLU),
        # (torch.nn.Linear, torch.nn.Hardtanh),
        # (torch.nn.Linear, torch.nn.GELU),
    ]

    assert len(patterns) == len(attr_names), "pattern and replacement length should be equal"
    for pattern, attr_name in zip(patterns, attr_names):
        for node in new_graph.nodes:
            print(node)
            
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
                    fused_linear = fuse_linear_eltwise_eval(linear, eltwise, attr_name, attr_names[attr_name])
                    replace_node_module(node.args[0], modules, fused_linear)
                    node.replace_all_uses_with(node.args[0])
                    new_graph.erase_node(node)
    
            elif matches_module_method_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of linear is used by other nodes
                    continue
                linear = modules[node.args[0].target]
                eltwise = node.target
                eval_mode = not linear.training

                tensors = example_inputs + [linear.weight]
                if linear.bias is not None:
                    tensors.append(linear.bias)
                is_cpu = all(x.device == torch.device('cpu') for x in tensors)
                if eval_mode and is_cpu:
                    fused_linear = fuse_linear_eltwise_eval(linear, eltwise, attr_name, attr_names[attr_name])
                    print("matches_module_method_pattern")
                    replace_node_module(node.args[0], modules, fused_linear)
                    node.replace_all_uses_with(node.args[0])
                    new_graph.erase_node(node)                                
    
    
    gm =  fx.GraphModule(gm, new_graph)    
    return gm