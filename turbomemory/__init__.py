#!/usr/bin/env python3
"""TurboMemory v0.4 - Enhanced layered memory system with quantization, verification, and observability."""

from .turbomemory import (
    TurboMemory,
    TurboMemoryConfig,
    ExclusionRules,
    QualityScore,
    VerificationResult,
    MemoryMetrics,
    compute_quality_score,
    quantize_packed,
    dequantize_packed,
    cosine_sim,
    now_iso,
    ensure_dir,
    safe_topic_filename,
    sha1_text,
    pack_unsigned,
    unpack_unsigned,
)

__version__ = "0.4.0"
__all__ = [
    "TurboMemory",
    "TurboMemoryConfig",
    "ExclusionRules",
    "QualityScore",
    "VerificationResult",
    "MemoryMetrics",
    "compute_quality_score",
    "quantize_packed",
    "dequantize_packed",
    "cosine_sim",
    "now_iso",
    "ensure_dir",
    "safe_topic_filename",
    "sha1_text",
    "pack_unsigned",
    "unpack_unsigned",
]
