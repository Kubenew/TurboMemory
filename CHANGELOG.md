# Changelog

All notable changes to TurboMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- TurboQuant as standalone reusable component (turbomemory/quantization.py)
- TMF (TurboMemory Format) v1 portable storage specification
- Hybrid search (BM25 + vector fusion)
- FastAPI server with multi-tenant support
- Log-based replication protocol (v0.8)
- Benchmark suite (turbomemory/benchmark.py)
- Comprehensive test coverage for core features

### Changed
- Modular package structure with clear separation of concerns
- SQLite PRAGMA optimizations (WAL mode, mmap, page_size)
- Updated pyproject.toml with proper dependencies and extras
- Expanded README with badges, benchmarks, and comparisons

### Fixed
- Type hints throughout codebase
- Exclusion rules for secret detection

---

## [0.5.0] - 2024-XX-XX

### Added
- **TurboQuant compression** - 4/6/8-bit packed embedding storage
- **TMF (TurboMemory Format)** - Portable storage format specification
- **Hybrid search** - BM25 + vector similarity fusion
- **FastAPI server** - Optional REST API with multi-tenant support
- **Replication protocol** - Log-based sync between nodes
- **Benchmark suite** - Compression ratios, latency, throughput tests
- **LangChain integration** - VectorStore and Retriever
- **LlamaIndex integration** - VectorStore adapter

### Changed
- **Storage format** - Topic-based partitioning with centroid prefilter
- **Self-healing** - Background consolidation daemon
- **Contradiction detection** - Automatic confidence decay

---

## [0.4.0] - 2024-XX-XX

### Added
- Topic-based storage with SQLite index
- Confidence decay and TTL expiration
- CLI commands (add_memory, query, stats, backup, restore)
- Streamlit dashboard

---

## [0.3.0] - 2024-XX-XX

### Added
- Core quantization (6-bit packed format)
- Basic vector search
- Append-only transcript logging
- Plugin system

---

## [0.2.0] - 2024-XX-XX

### Added
- Initial Python package structure
- Sentence-transformers integration
- Basic CLI

---

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- Concept prototype
