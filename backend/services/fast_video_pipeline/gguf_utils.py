# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportUnusedImport=false, reportMissingImports=false, reportUnusedVariable=false, reportConstantRedefinition=false, reportUnboundVariable=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingParameterType=false, reportIncompatibleMethodOverride=false, reportPropertyTypeMismatch=false, reportIncompatibleVariableOverride=false
import contextlib
import logging
from typing import Any, Generator

import gguf
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps
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

    @property  # pyright: ignore[reportIncompatibleMethodOverride, reportPropertyTypeMismatch]
    def shape(self) -> Any:
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


# Store the original Linear class at import time so the monkey-patch
# in gguf_linear_context() can always restore it.
_OriginalLinear = torch.nn.Linear


class GGUFLinear(_OriginalLinear):  # type: ignore[misc]
    """Drop-in replacement for ``nn.Linear`` that keeps quantised GGUF tensors
    and dequantises them on the fly during forward.  Inheriting from the real
    ``nn.Linear`` ensures ``isinstance`` checks pass everywhere."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        # Call Module.__init__ directly – we do NOT want Linear's __init__
        # to create real weight/bias Parameters (they come from the GGUF
        # state dict via _load_from_state_dict later).
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        # Create meta-device placeholders so that reconcile_state_dict()
        # can see the expected shapes and slice oversized GGUF tensors.
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device="meta"), requires_grad=False
        )
        self.bias = (
            torch.nn.Parameter(torch.empty(out_features, device="meta"), requires_grad=False)
            if bias else None
        )

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
    def metadata(self, path: str) -> dict[str, Any]:
        """Read model config from the GGUF file's embedded metadata.

        GGUF files created for LTX-Video (e.g. by Kijai/unsloth)
        embed the full model config as a JSON string in the ``config``
        metadata field.  We parse that directly so the builder
        constructs the *exact* architecture that matches the weights.
        """
        reader = gguf.GGUFReader(path)
        config_field = reader.get_field("config")
        if config_field is not None:
            import json
            config_str = str(config_field.parts[config_field.data[-1]], encoding="utf-8")
            config: dict[str, Any] = json.loads(config_str)
            logger.info("Loaded model config from GGUF metadata")
            return config

        # Fallback: if the GGUF has no embedded config, use a reasonable
        # default for the LTX-Video 22B architecture.
        logger.warning("GGUF file has no embedded config metadata, using hardcoded fallback")
        return {
            "transformer": {
                "activation_fn": "gelu-approximate",
                "apply_gated_attention": False,
                "attention_bias": True,
                "attention_head_dim": 128,
                "attention_type": "default",
                "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048,
                "audio_in_channels": 128,
                "audio_num_attention_heads": 32,
                "audio_out_channels": 128,
                "audio_positional_embedding_max_pos": [20],
                "av_ca_timestep_scale_multiplier": 1,
                "av_cross_ada_norm": True,
                "caption_channels": 4096,
                "caption_proj_before_connector": True,
                "cross_attention_adaln": True,
                "cross_attention_dim": 4096,
                "cross_attention_norm": True,
                "double_self_attention": False,
                "dropout": 0.0,
                "in_channels": 128,
                "norm_elementwise_affine": False,
                "norm_eps": 1e-06,
                "num_attention_heads": 32,
                "num_embeds_ada_norm": 1000,
                "num_layers": 48,
                "only_cross_attention": False,
                "out_channels": 128,
                "positional_embedding_max_pos": [20, 2048, 2048],
                "positional_embedding_theta": 10000.0,
                "positional_embedding_type": "rope",
                "qk_norm": "rms_norm",
                "rope_type": "interleaved",
                "share_ff": False,
                "standardization_norm": "rms_norm",
                "timestep_scale_multiplier": 1000,
                "upcast_attention": False,
                "use_audio_video_cross_attention": True,
                "use_linear_projection": False,
                "use_middle_indices_grid": True
            }
        }

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        if isinstance(path, list):
            path = path[0]
            
        logger.info(f"Loading GGUF checkpoing from: {path}")
        reader = gguf.GGUFReader(path)
        sd: dict[str, Any] = {}
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


def reconcile_state_dict(
    sd: dict[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Generically slice oversized tensors in ``sd`` so they match ``model``.

    For every key present in both the state dict and the model's
    ``state_dict()``, if the loaded tensor is *larger* than the model
    parameter on any dimension, it is sliced (from the start) to fit.
    Tensors that are already the correct size – or smaller – are left
    untouched.

    This keeps the fix future-proof: no key names are hard-coded.
    """
    model_sd = model.state_dict()
    reconciled: dict[str, Any] = {}
    for key, value in sd.items():
        if key in model_sd and hasattr(value, "shape") and hasattr(model_sd[key], "shape"):
            # Skip quantized tensors – their packed data layout means
            # normal slicing would corrupt the values.
            if is_quantized(value):
                reconciled[key] = value
                continue
            target_shape = model_sd[key].shape
            source_shape = value.shape
            if len(target_shape) == len(source_shape):
                slices: list[slice] = []
                needs_slice = False
                for src_dim, tgt_dim in zip(source_shape, target_shape):
                    if src_dim > tgt_dim:
                        slices.append(slice(0, tgt_dim))
                        needs_slice = True
                    else:
                        slices.append(slice(None))
                if needs_slice:
                    logger.info(
                        "Slicing GGUF tensor %s from %s to %s",
                        key,
                        list(source_shape),
                        list(target_shape),
                    )
                    value = value[tuple(slices)]
        reconciled[key] = value
    return reconciled


@contextlib.contextmanager
def gguf_linear_context() -> Generator[None, None, None]:
    """Context manager that monkey-patches ``torch.nn.Linear`` → ``GGUFLinear``.

    Use this around model construction so that every ``nn.Linear`` created
    inside the ``with`` block is actually a ``GGUFLinear``.  This avoids the
    stale-reference problem: modules that cache references to Linear layers
    (like ``TransformerArgsPreprocessor.patchify_proj``) will hold references
    to ``GGUFLinear`` objects from the start.
    """
    torch.nn.Linear = GGUFLinear  # type: ignore[misc]
    try:
        yield
    finally:
        torch.nn.Linear = _OriginalLinear  # type: ignore[misc]
