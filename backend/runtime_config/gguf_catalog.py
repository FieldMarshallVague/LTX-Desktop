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
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "filename": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
        "expected_size_bytes": 14_326_857_120,
        "vram_required_gb": 13.0,
        "is_text_encoder": False,
        "is_distilled": True,
    },
    "ltx-2.3-q5_k_s-unsloth": {
        "id": "ltx-2.3-q5_k_s-unsloth",
        "name": "LTX-2.3 Q5_K_S (unsloth)",
        "description": "Distilled LTX-2.3 model. Recommended ~15GB VRAM. Excellent quality.",
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "filename": "distilled/ltx-2.3-22b-distilled-Q5_K_S.gguf",
        "expected_size_bytes": 15_248_901_536,
        "vram_required_gb": 15.0,
        "is_text_encoder": False,
        "is_distilled": True,
    },
    "ltx-2.3-q6_k-unsloth": {
        "id": "ltx-2.3-q6_k-unsloth",
        "name": "LTX-2.3 Q6_K (unsloth)",
        "description": "Distilled LTX-2.3 model. Recommended ~18GB VRAM. Near-original quality.",
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "filename": "distilled/ltx-2.3-22b-distilled-Q6_K.gguf",
        "expected_size_bytes": 17_774_906_784,
        "vram_required_gb": 18.0,
        "is_text_encoder": False,
        "is_distilled": True,
    },
    "ltx-2.3-q8_0-unsloth": {
        "id": "ltx-2.3-q8_0-unsloth",
        "name": "LTX-2.3 Q8_0 (unsloth)",
        "description": "Distilled LTX-2.3 model. Recommended ~22GB VRAM. Near-perfect precision.",
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "filename": "distilled/ltx-2.3-22b-distilled-Q8_0.gguf",
        "expected_size_bytes": 22_755_540_384,
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
    },
    "gemma-3-12b-it-q4_k_m": {
        "id": "gemma-3-12b-it-q4_k_m",
        "name": "Gemma 3 12B Q4_K_M (unsloth)",
        "description": "Quantized Gemma 3 text encoder. ~7-8GB VRAM footprint.",
        "repo_id": "unsloth/gemma-3-12b-it-GGUF",
        "filename": "gemma-3-12b-it-Q4_K_M.gguf",
        "expected_size_bytes": 7_300_778_336,
        "vram_required_gb": 8.0,
        "is_text_encoder": True,
        "is_distilled": False,
    },
    "gemma-3-12b-it-q5_k_m": {
        "id": "gemma-3-12b-it-q5_k_m",
        "name": "Gemma 3 12B Q5_K_M (unsloth)",
        "description": "Quantized Gemma 3 text encoder. ~9GB VRAM footprint. Higher quality.",
        "repo_id": "unsloth/gemma-3-12b-it-GGUF",
        "filename": "gemma-3-12b-it-Q5_K_M.gguf",
        "expected_size_bytes": 8_445_036_896,
        "vram_required_gb": 9.0,
        "is_text_encoder": True,
        "is_distilled": False,
    }
}
