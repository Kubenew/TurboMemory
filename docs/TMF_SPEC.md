# TMF v1 — TurboMemory Format Specification

**Version**: 1.0 (Draft)  
**Status**: In Progress (Target: v0.5)  
**Date**: April 2026  
**Purpose**: Define a portable, verifiable, and efficient on-disk format for semantic memory that combines SQLite simplicity with TurboQuant-compressed embeddings.

## 1. Design Goals

- **Portability** — The entire memory store must be a self-contained directory that can be copied, zipped, backed up, or moved between machines/devices with zero reconfiguration.
- **Durability** — Append-only log as the single source of truth; all other files are derived/cache.
- **Efficiency** — Extreme compression via TurboQuant (4/6/8-bit packed scalar quantization) while preserving high cosine similarity.
- **Simplicity** — Leverage SQLite for rich metadata and fast filtering.
- **Verifiability** — Checksums everywhere; fast validation on load.
- **Extensibility** — Versioned, with clear migration path.
- **Topic-aware** — Support centroid-based pre-filtering and on-demand loading.

## 2. On-Disk Layout

A TurboMemory store is a directory (default: `./tm_data/` or user-specified root):
