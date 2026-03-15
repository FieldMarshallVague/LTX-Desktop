"""Catalog of available community GGUF models (Kijai, unsloth, etc)."""

from __future__ import annotations

from typing import TypedDict, Literal

GgufModelId = str

class GgufModelDef(TypedDict):
    id: GgufModelId
    name: str
    description: str
    repo_id: str
    filename: str
    expected_size_bytes: int
    vram_required_gb: float
    is_text_encoder: bool
    is_distilled: bool  # If False, the UI/backend will know to attach the distilled LoRA

GGUF_CATALOG: dict[GgufModelId, GgufModelDef] = {
    "ltx-2.3-q4_k_m-unsloth": {
        "id": "ltx-2.3-q4_k_m-unsloth",
        "name": "LTX-2.3 Q4_K_M (unsloth)",
        "description": "Distilled LTX-2.3 model. Recommended ~12-13GB VRAM. Fast and good quality.",
        "repo_id": "unsloth/LTX-Video-GGUF",
        "filename": "ltx-2.3-q4_k_m.gguf",
        "expected_size_bytes": 13_100_000_000,
        "vram_required_gb": 13.0,
        "is_text_encoder": False,
        "is_distilled": True,
    },
    "ltx-2.3-q5_k_s-unsloth": {
        "id": "ltx-2.3-q5_k_s-unsloth",
        "name": "LTX-2.3 Q5_K_S (unsloth)",
        "description": "Distilled LTX-2.3 model. Recommended ~15GB VRAM. Excellent quality.",
        "repo_id": "unsloth/LTX-Video-GGUF",
        "filename": "ltx-2.3-q5_k_s.gguf",
        "expected_size_bytes": 15_200_000_000,
        "vram_required_gb": 15.0,
        "is_text_encoder": False,
        "is_distilled": True,
    },
    "ltx-2.3-q6_k-unsloth": {
        "id": "ltx-2.3-q6_k-unsloth",
        "name": "LTX-2.3 Q6_K (unsloth)",
        "description": "Distilled LTX-2.3 model. Recommended ~18GB VRAM. Near-original quality.",
        "repo_id": "unsloth/LTX-Video-GGUF",
        "filename": "ltx-2.3-q6_k.gguf",
        "expected_size_bytes": 17_800_000_000,
        "vram_required_gb": 18.0,
        "is_text_encoder": False,
        "is_distilled": True,
    },
    "ltx-2.3-q8_0-unsloth": {
        "id": "ltx-2.3-q8_0-unsloth",
        "name": "LTX-2.3 Q8_0 (unsloth)",
        "description": "Distilled LTX-2.3 model. Recommended ~22GB VRAM. Near-perfect precision.",
        "repo_id": "unsloth/LTX-Video-GGUF",
        "filename": "ltx-2.3-q8_0.gguf",
        "expected_size_bytes": 22_800_000_000,
        "vram_required_gb": 22.0,
        "is_text_encoder": False,
        "is_distilled": True,
    },
    "gemma-2-9b-it-q4_k_m": {
        "id": "gemma-2-9b-it-q4_k_m",
        "name": "Gemma 2 9B Q4_K_M (city96)",
        "description": "Quantized text encoder. Reduces text encoder VRAM heavily.",
        "repo_id": "city96/gemma-2-9b-it-gguf",
        "filename": "gemma-2-9b-it-q4_k_m.gguf",
        "expected_size_bytes": 5_800_000_000,
        "vram_required_gb": 6.0,
        "is_text_encoder": True,
        "is_distilled": False,
    },
    "gemma-2-9b-it-q5_k_m": {
        "id": "gemma-2-9b-it-q5_k_m",
        "name": "Gemma 2 9B Q5_K_M (city96)",
        "description": "Quantized text encoder. Slightly better quality text processing.",
        "repo_id": "city96/gemma-2-9b-it-gguf",
        "filename": "gemma-2-9b-it-q5_k_m.gguf",
        "expected_size_bytes": 6_800_000_000,
        "vram_required_gb": 7.0,
        "is_text_encoder": True,
        "is_distilled": False,
    }
}
