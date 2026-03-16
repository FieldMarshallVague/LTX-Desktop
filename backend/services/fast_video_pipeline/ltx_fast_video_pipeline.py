"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Any, cast

import torch
import logging

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

logger = logging.getLogger(__name__)


class LTXFastVideoPipeline:
    pipeline_kind: str

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        distilled_lora_path: str | None = None,
        model_type: str = "fast",
        gguf_path: str | None = None,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            distilled_lora_path=distilled_lora_path,
            model_type=model_type,
            gguf_path=gguf_path,
        )

    def __init__(
        self, 
        checkpoint_path: str, 
        gemma_root: str | None, 
        upsampler_path: str, 
        device: torch.device,
        distilled_lora_path: str | None = None,
        model_type: str = "fast",
        gguf_path: str | None = None,
    ) -> None:
        self.pipeline_kind = model_type
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        loras = [distilled_lora_path] if distilled_lora_path else []

        is_gguf = gguf_path is not None
        quant_policy = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) and not is_gguf else None

        logger.debug("Initializing DistilledPipeline: checkpoint=%s, gemma=%s, upsampler=%s, is_gguf=%s, gguf_path=%s", 
                     checkpoint_path, gemma_root, upsampler_path, is_gguf, gguf_path)
        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=cast(Any, loras),
            device=device,
            quantization=quant_policy,
        )
        logger.debug("DistilledPipeline initialized successfully")

        if is_gguf:
            from dataclasses import replace
            from services.fast_video_pipeline.gguf_utils import GgufModelStateDictLoader, gguf_linear_context, reconcile_state_dict
            from ltx_core.model.transformer import X0Model
            # Swap only the transformer builder to use the GGUF file.
            # All other builders (VAE, audio, vocoder, text encoder) keep
            # using the standard safetensors checkpoint passed above.
            tb = self.pipeline.model_ledger.transformer_builder
            gguf_builder = replace(
                tb,
                model_path=gguf_path,
                model_loader=GgufModelStateDictLoader(),
                model_sd_ops=None,
            )
            self.pipeline.model_ledger.transformer_builder = gguf_builder

            # Override the transformer() method to:
            # 1. Construct the meta model inside gguf_linear_context() so
            #    every nn.Linear is actually a GGUFLinear from the start -
            #    this avoids stale references in preprocessors.
            # 2. Reconcile oversized GGUF tensors to match the model.
            # 3. Zero-init any parameters missing from the GGUF.
            ledger = self.pipeline.model_ledger
            _original_target_device = ledger._target_device  # pyright: ignore[reportPrivateUsage]

            def _gguf_transformer() -> X0Model:
                builder = ledger.transformer_builder
                device = _original_target_device()
                config = cast(dict[str, Any], builder.model_config())  # pyright: ignore[reportUnknownMemberType]
                # Build meta model with GGUFLinear layers everywhere.
                with gguf_linear_context():
                    meta_model = builder.meta_model(config, builder.module_ops)  # pyright: ignore[reportUnknownMemberType]
                model_paths = list(builder.model_path) if isinstance(builder.model_path, tuple) else [builder.model_path]
                model_state_dict = builder.load_sd(model_paths, sd_ops=builder.model_sd_ops, registry=builder.registry, device=device)
                sd = reconcile_state_dict(cast(dict[str, Any], model_state_dict.sd), meta_model)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                meta_model.load_state_dict(sd, strict=False, assign=True)
                # Zero-init any parameters/buffers still on meta device
                # (e.g. caption_projection weights absent from the GGUF).
                for name, param in list(meta_model.named_parameters()):
                    if str(param.device) == "meta":
                        logger.info("Zero-initializing missing GGUF param: %s %s", name, list(param.shape))
                        setattr(
                            _resolve_parent(meta_model, name),
                            name.rsplit(".", 1)[-1],
                            torch.nn.Parameter(torch.zeros(param.shape, device=device), requires_grad=False),
                        )
                for name, buf in list(meta_model.named_buffers()):
                    if str(buf.device) == "meta":
                        logger.info("Zero-initializing missing GGUF buffer: %s %s", name, list(buf.shape))
                        parent = _resolve_parent(meta_model, name)
                        parent.register_buffer(name.rsplit(".", 1)[-1], torch.zeros(buf.shape, device=device))
                return X0Model(meta_model).to(ledger.device).eval()

            def _resolve_parent(root: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
                """Return the parent module for a dotted parameter name."""
                parts = dotted_name.rsplit(".", 1)
                if len(parts) == 1:
                    return root
                parent_path = parts[0]
                mod = root
                for attr in parent_path.split("."):
                    mod = getattr(mod, attr)
                return mod

            # Bind the custom builder as the transformer() method.
            import types
            ledger.transformer = types.MethodType(lambda self: _gguf_transformer(), ledger)  # type: ignore[assignment]

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        logger.debug("Running inference: seed=%d, size=%dx%d, num_frames=%d, num_images=%d", 
                     seed, width, height, num_frames, len(images))
        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        transformer = self.pipeline.model_ledger.transformer()

        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),  # type: ignore[reportUnknownMemberType]
        )
        setattr(self.pipeline.model_ledger, "transformer", lambda: compiled)
