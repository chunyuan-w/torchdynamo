import copy
import itertools
import logging
import random
import weakref

import torch
import torch.nn as nn
from torch import _prims
from torch.fx.experimental.optimization import matches_module_pattern
from torch.fx.experimental.optimization import replace_node_module
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode

log = logging.getLogger(__name__)


class AutogradMonkeypatch(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        if func is replacements:
            return replacements[func](*args, **kwargs)
        return func(*args, **kwargs)


patch_functions = AutogradMonkeypatch


def replace_fx(gm: torch.fx.GraphModule):
    # Sometimes patch_functions() misses things already in the graph
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in replacements:
            with gm.graph.inserting_before(node):
                node.replace_all_uses_with(
                    gm.graph.call_function(
                        replacements[node.target], node.args, node.kwargs
                    )
                )
            gm.graph.erase_node(node)
    gm.recompile()
    return gm


class EltwiseFusionOp:
    def __init__(self, post_op, scalars=[], algorithm=""):
        self.post_op = post_op
        self.scalars = scalars
        self.algorithm = algorithm


class LinearEltwise(nn.Linear):
    def __init__(
        self,
        linear,
        eltwise,
        op_name,
        op_info,
        in_features,
        out_features,
        bias,
        device,
        dtype,
    ):
        super(LinearEltwise, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._update_module_params(linear, eltwise, op_name, op_info)

    def _update_module_params(self, linear, eltwise, op_name, op_info):
        self.__dict__ = copy.deepcopy(linear.__dict__)

        self.attr = op_name

        assert all(hasattr(eltwise, item) for item in op_info.scalars)
        self.scalars = [getattr(eltwise, item) for item in op_info.scalars]

        algorithm = ""
        if op_info.algorithm:
            assert hasattr(eltwise, op_info.algorithm)
            algorithm = getattr(eltwise, op_info.algorithm)
        self.algorithm = algorithm

    def forward(self, input):
        y = torch.ops.mkldnn_prepacked.linear_eltwise(
            input, self.weight, self.bias, self.attr, self.scalars, self.algorithm
        )
        return y


def fuse_linear_eltwise_eval(linear, eltwise, op_name, op_info):
    return LinearEltwise(
        linear,
        eltwise,
        op_name,
        op_info,
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        linear.weight.device,
        linear.weight.dtype,
    )


def replace_functional(gm: torch.fx.GraphModule):
    for node in reversed(list(gm.graph.nodes)):
        if (
            node.op == "call_function" or node.op == "call_method"
        ) and node.target in functional_to_module:
            # TODO: handle args
            if len(node.args) != 1:
                continue
            module_name = "%s_module" % node.name
            module = functional_to_module.get(node.target)(**node.kwargs)
            gm.add_submodule(module_name, module)

            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_module(module_name=module_name, args=node.args)
                node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            dict(gm.named_modules())[module_name].training = gm.training
    gm.recompile()
    return gm


def fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu") for example_input in example_inputs
    )
    if not is_cpu:
        return gm
    gm = replace_functional(gm)

    modules = dict(gm.named_modules())

    for (pointwise_name, pointwise_info), (
        computation_name,
        fuse_func,
    ) in itertools.product(pointwise_op_map.items(), computation_op_map.items()):
        pattern = (computation_name, pointwise_info.post_op)
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if (
                    len(node.args[0].users) > 1
                ):  # Output of linear is used by other nodes
                    continue
                linear = modules[node.args[0].target]
                eltwise = modules[node.target]
                eval_mode = all(not n.training for n in [linear, eltwise])
                if not eval_mode:
                    continue
                fused_linear = fuse_func(
                    linear, eltwise, pointwise_name, pointwise_info
                )
                replace_node_module(node.args[0], modules, fused_linear)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
    gm.recompile()
    return gm


def _philox_rand_like_meta(input, seed, offset):
    return _prims.TensorMeta(input)


def _philox_rand_like(input, seed, offset):
    # placeholder only used in tracing
    return torch.rand_like(input)


philox_rand_like = _prims._make_prim(
    schema="philox_rand_like(Tensor input, Tensor seed, int offset) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_rand_like_meta,
    impl_aten=_philox_rand_like,
    doc="",
)


def _philox_seed_like_meta(x):
    return _prims.TensorMeta(_philox_seed_like(x))


def _philox_seed_like(x):
    # we need a tensor input here so AOT autograd properly captures this
    # with just a device input, this becomes a constant
    return torch.tensor(random.randrange(2**31), device=x.device, dtype=torch.int32)


philox_seed_like = _prims._make_prim(
    schema="philox_seed_like(Tensor other) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_seed_like_meta,
    impl_aten=_philox_seed_like,
    doc="",
)


def null_ref():
    return None


class PhiloxRandomState:
    next_offset = 0
    seed = {}
    last_tracer_ref = null_ref

    @classmethod
    def reset(cls, tracer=None):
        cls.next_offset = 0
        cls.seed = {}
        cls.last_tracer_ref = weakref.ref(tracer) if tracer is not None else null_ref

    @classmethod
    def get_seed_offset(cls, x):
        modes = torch.fx.experimental.proxy_tensor.get_torch_dispatch_modes()
        proxy_modes = [m for m in modes if isinstance(m, ProxyTorchDispatchMode)]
        if proxy_modes:
            tracer = proxy_modes[0].tracer
            if cls.last_tracer_ref() is not tracer:
                # tracer changed, need to reset state
                cls.reset(tracer)
        else:
            # no tracer, need to reset state
            cls.reset()

        device = x.device
        if device not in cls.seed:
            # Compute the seed just once per trace so that we pass fewer
            # things from forward to backward
            cls.seed[device] = philox_seed_like(x)

        seed = cls.seed[device]
        offset = cls.next_offset
        cls.next_offset += x.numel()
        return seed, offset


class LowmemDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        scale = float(1.0 / (1.0 - p))
        seed, offset = PhiloxRandomState.get_seed_offset(x)
        ctx.save_for_backward(seed)
        ctx.offset = offset
        bool_mask = philox_rand_like(x, seed, offset) > p
        return bool_mask.to(x.dtype) * x * scale

    @staticmethod
    def backward(ctx, grad_output):
        p = ctx.p
        scale = float(1.0 / (1.0 - p))
        (seed,) = ctx.saved_tensors
        bool_mask = philox_rand_like(grad_output, seed, ctx.offset) > p
        return bool_mask.to(grad_output.dtype) * grad_output * scale, None


@torch.fx.wrap
def lowmem_dropout(input, p, training=True, inplace=False):
    if isinstance(input, torch.fx.Proxy):
        # double check we don't FX trace this
        return input.tracer.create_proxy(
            "call_function",
            lowmem_dropout,
            (input, p, training),
            {},
        )
    if not training or p == 0:
        return input
    result = LowmemDropout.apply(input, p)
    if inplace:
        input.copy_(result)
    return result


@torch.fx.wrap
def rand_like(x, **kwargs):
    if isinstance(x, torch.fx.Proxy):
        # double check we don't FX trace this
        return x.tracer.create_proxy("call_function", rand_like, (x), kwargs)
    assert kwargs.get("device", x.device) == x.device
    seed, offset = PhiloxRandomState.get_seed_offset(x)
    return philox_rand_like(x, seed, offset).to(kwargs.get("dtype", torch.float32))


replacements = {torch.nn.functional.dropout: lowmem_dropout, torch.rand_like: rand_like}

computation_op_map = {nn.Linear: fuse_linear_eltwise_eval}

pointwise_op_map = {
    "relu": EltwiseFusionOp(nn.ReLU),
    "sigmoid": EltwiseFusionOp(nn.Sigmoid),
    "tanh": EltwiseFusionOp(nn.Tanh),
    "hardswish": EltwiseFusionOp(nn.Hardswish),
    "leaky_relu": EltwiseFusionOp(nn.LeakyReLU, scalars=["negative_slope"]),
    "hardtanh": EltwiseFusionOp(nn.Hardtanh, scalars=["min_val", "max_val"]),
    "gelu": EltwiseFusionOp(nn.GELU, algorithm="approximate"),
}

functional_to_module = {
    torch.relu: nn.ReLU,
    torch.nn.functional.gelu: nn.GELU,
}
