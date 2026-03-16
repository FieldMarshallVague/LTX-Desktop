import logging
from typing import Any

import gguf
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.module_ops import ModuleOps
from services.fast_video_pipeline.dequant import dequantize_tensor, is_quantized

logger = logging.getLogger(__name__)

class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type=None, tensor_shape=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type=None, tensor_shape=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


class GGUFLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight = state_dict.get(f"{prefix}weight")
        bias = state_dict.get(f"{prefix}bias")
        
        if weight is None:
            missing_keys.append(f"{prefix}weight")
            v = torch.zeros(self.out_features, self.in_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
        else:
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        elif hasattr(self, "bias") and self.bias is not None:
             self.bias = torch.nn.Parameter(torch.zeros(self.out_features), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if is_quantized(weight):
            weight = dequantize_tensor(weight, dtype=input.dtype)
        else:
            weight = weight.to(input.dtype)
            
        bias = self.bias
        if bias is not None:
            bias = bias.to(input.dtype)
            
        return torch.nn.functional.linear(input, weight, bias)


class GgufModelStateDictLoader(StateDictLoader):
    def metadata(self, path: str) -> dict:
        return {}

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        if isinstance(path, list):
            path = path[0]
            
        logger.info(f"Loading GGUF checkpoing from: {path}")
        reader = gguf.GGUFReader(path)
        sd = {}
        size: int = 0
        dtypes = set()
        device_to_use = device or torch.device("cpu")
        
        handle_prefix = "model.diffusion_model."
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

        tensors = []
        for tensor in reader.tensors:
            sd_key = tensor.name
            if has_prefix:
                if not tensor.name.startswith(handle_prefix):
                    continue
                sd_key = tensor.name[prefix_len:]
                
            expected_name: str | None = None
            if sd_ops is None:
                expected_name = sd_key
            else:
                expected_name = sd_ops.apply_to_key(sd_key)
                
            if expected_name is None:
                continue
            
            # The numpy array needs to not be writable to avoid warnings
            torch_tensor = torch.from_numpy(tensor.data)
            
            shape = None
            field_key = f"comfy.gguf.orig_shape.{tensor.name}"
            field = reader.get_field(field_key)
            if field is not None:
                shape = torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))
            else:
                shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
                
            if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                torch_tensor = torch_tensor.view(*shape)
                
            mapped_tensor = GGMLTensor(
                torch_tensor, 
                tensor_type=tensor.tensor_type, 
                tensor_shape=shape
            )
            
            if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                mapped_tensor = dequantize_tensor(mapped_tensor, dtype=torch.float32)

            key_value_pairs = ((expected_name, mapped_tensor),)
            if sd_ops is not None:
                key_value_pairs = sd_ops.apply_to_key_value(expected_name, mapped_tensor)
                
            for key, value in key_value_pairs:
                if hasattr(value, "element_size") and hasattr(value, "nelement"):
                    import typing
                    el_size = typing.cast(int, getattr(value, "element_size")())
                    nel = typing.cast(int, getattr(value, "nelement")())
                    size += el_size * nel
                if hasattr(value, "dtype"):
                    dtypes.add(value.dtype)
                sd[key] = value

        return StateDict(sd=sd, device=device_to_use, size=size, dtype=dtypes)


def is_linear(module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Linear)

def swap_linear_with_gguf(module: torch.nn.Module) -> torch.nn.Module:
    new_module = GGUFLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None
    )
    return new_module

gguf_module_ops = ModuleOps(matcher=is_linear, mutator=swap_linear_with_gguf)
