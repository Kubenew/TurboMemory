# TurboMemory ⚡

[![PyPI Version](https://img.shields.io/pypi/v/turbomemory)](https://pypi.org/project/turbomemory/)
[![Python Versions](https://img.shields.io/pypi/pyversions/turbomemory)](https://pypi.org/project/turbomemory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Kubenew/TurboMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/Kubenew/TurboMemory/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/Kubenew/TurboMemory)](https://codecov.io/gh/Kubenew/TurboMemory)
[![Downloads](https://img.shields.io/pypi/dm/turbomemory)](https://pypi.org/project/turbomemory/)

**TurboMemory is a lightweight semantic storage engine for compressed embedding archives.**

It combines:
- **SQLite metadata indexing**
- **append-only transcript logging**
- **quantized embedding storage** (4-bit / 6-bit / 8-bit packed format)
- **topic-based partitioning + centroid prefiltering**
- **background consolidation** (merge / prune / deduplicate)
- optional **confidence decay** + **contradiction detection**

TurboMemory is designed for **local-first semantic search**, offline RAG, and edge deployments.

> Goal: deliver "SQLite simplicity" for semantic memory + compressed vector storage.

---

## Why TurboMemory?

Embedding storage is expensive:
- float32 vectors consume large disk space
- most vector DBs are heavy to deploy
- local-first apps need portable storage formats

TurboMemory solves this by using **TurboQuant-style packing** to store embeddings efficiently while still enabling fast retrieval.

---

## Features

### Storage
- Append-only transcript/event log (durable ingestion)
- Topic-based storage files (load-on-demand)
- SQLite index for metadata + fast filtering
- Packed embedding formats: **4-bit / 6-bit / 8-bit**

### Retrieval
- centroid/topic prefilter to reduce search space
- configurable scoring pipeline
- optional verification filtering

### Maintenance / Self-Healing
- background consolidation daemon
- deduplication and merging of similar chunks
- TTL expiration + confidence decay
- experimental contradiction detection

---

## Installation

### From PyPI (recommended)
```bash
pip install turbomemory
```

### From source
```bash
git clone https://github.com/Kubenew/TurboMemory.git
cd TurboMemory
pip install -e .
```

### With all features
```bash
pip install turbomemory[all]
```

### Requirements
- Python 3.9+
- numpy >= 1.24.0
- sentence-transformers >= 2.2.0

---

## Quickstart

### CLI Usage

```bash
# Add memory
python -m turbomemory add_memory --topic notes --text "TurboMemory stores semantic chunks efficiently."

# Query
python -m turbomemory query --query "semantic storage" --k 5

# Get stats
python -m turbomemory stats
```

### Python Usage

```python
from turbomemory import TurboMemory

tm = TurboMemory(root="./tm_data")

# Add memory
tm.add_memory(
    topic="notes",
    text="TurboMemory stores semantic chunks efficiently.",
    ttl_days=365
)

# Query
results = tm.query("semantic storage", k=5)

for score, topic, chunk in results:
    print(f"[{score:.3f}] {chunk['text']}")
```

**Example output:**
```
[0.892] TurboMemory stores semantic chunks efficiently.
[0.756] Semantic search with compression
[0.723] Vector storage made simple
```

---

## CLI Command Reference

| Command | Description |
|---------|-------------|
| `add_memory` | Add a memory chunk |
| `add_turn` | Add conversation turn |
| `query` | Search memories |
| `stats` | Show statistics |
| `backup` | Create backup |
| `restore` | Restore from backup |
| `export` | Export topics |
| `import` | Bulk import |
| `merge` | Merge topics |
| `sync` | Sync with remote |
| `hybrid` | Hybrid search |

See `python -m turbomemory --help` for full options.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TurboMemory                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  CLI / API    │  │ Python SDK    │  │ Integrations │       │
│  └──────┬─────────┘  └──────┬─────────┘  └──────┬─────────┘       │
│         │                   │                   │                  │
│  ┌──────▼───────────────────▼───────────────────▼─────────┐    │
│  │                    Core Engine                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │    │
│  │  │ Quantization│  │   Search    │  │ Consolidation│    │    │
│  │  │  (4/6/8bit)│  │  (BM25+Vec) │  │   Daemon    │    │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │    │
│  └─────────┼────────────────┼────────────────┼────────────┘    │
│            │                │                │                 │
│  ┌─────────▼────────────────▼────────────────▼────────────┐  │
│  │                    Storage Layer                       │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │  │SQLite   │  │  TMF    │  │  .tmlog │  │  Sync   │   │  │
│  │  │Index    │  │ Vectors │  │   Log   │  │ Protocol│   │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Benchmarks

### Compression Ratios

| Format | Size (10K vectors, 384 dims) | Compression |
|--------|-------------------------------|-------------|
| float32 | 14.6 MB | 1x |
| 8-bit | 3.7 MB | **4x** |
| 6-bit | 2.8 MB | **5.2x** |
| 4-bit | 1.8 MB | **8x** |

### Query Latency

| Dataset Size | Latency (P95) |
|--------------|---------------|
| 1,000 chunks | 12ms |
| 10,000 chunks | 45ms |
| 100,000 chunks | 180ms |

### Recall Quality

| Bit Depth | Avg Cosine Similarity |
|-----------|----------------------|
| 8-bit | 0.997 |
| 6-bit | 0.968 |
| 4-bit | 0.912 |

Run benchmarks yourself:
```bash
python -m turbomemory.benchmark
```

---

## Comparison

| Feature | TurboMemory | Chroma | sqlite-vector | LanceDB |
|---------|-------------|--------|---------------|---------|
| Compression | 4-8x | None | None | None |
| Local-first | ✅ | ❌ | ✅ | ✅ |
| SQLite backend | ✅ | ❌ | ✅ | ❌ |
| Topic partitioning | ✅ | ❌ | ❌ | ❌ |
| Self-healing | ✅ | ❌ | ❌ | ❌ |
| Replication | ✅ | ❌ | ❌ | ✅ |
| Hybrid search | ✅ | ✅ | ❌ | ✅ |
| No server needed | ✅ | ❌ | ✅ | ❌ |

---

## Integrations

### LangChain
```python
from turbomemory.integrations import TurboMemoryVectorStore

vectorstore = TurboMemoryVectorStore(root="./data", topic="docs")
vectorstore.add_texts(["doc1", "doc2"])
docs = vectorstore.similarity_search("query")
```

### LlamaIndex
```python
from turbomemory.integrations import getTurboMemoryIndex

index = getTurboMemoryIndex(root="./data")
query_engine = index.as_query_engine()
response = query_engine.query("your question")
```

---

## Limitations

- **No distributed clustering** - Designed for single-node deployment
- **No real-time multi-writer** - Single-writer with eventual consistency via sync
- **HNSW/IVF not default** - Uses centroid prefilter; optional HNSW available
- **Model pinned at ingest** - All vectors must use same embedding model

---

## Glossary

- **Centroid prefilter**: Pre-selects relevant topics using centroid similarity before full search
- **Confidence decay**: Reduces confidence of older memories over time
- **Contradiction detection**: Detects conflicting information and adjusts confidence
- **Consolidation**: Background process to merge/prune/optimize storage
- **TurboQuant**: 4/6/8-bit packed quantization for embeddings
- **TMF**: [TurboMemory Format](docs/TMF_SPEC.md) - portable storage format specification

---

## Roadmap

See [ROADMAP.md](ROADMAP.md)

| Version | Milestone |
|---------|-----------|
| v0.3 | Stability + CI + packaging |
| v0.4 | Benchmarks + profiling |
| v0.5 | TMF v1 stable format |
| v0.6 | Hybrid search (BM25 + vector) |
| v0.7 | FastAPI server mode |
| v0.8 | Replication / edge sync |

---

## Docker

```bash
# Build
docker build -t turbomemory .

# Run
docker run -p 8000:8000 turbomemory

# Or use docker-compose
docker compose up
```

---

## Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Run linters: `ruff check . && black .`
5. Submit a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Support

- 📖 [Documentation](https://github.com/Kubenew/TurboMemory#readme)
- 🐛 [Issue Tracker](https://github.com/Kubenew/TurboMemory/issues)
- 💬 [Discussions](https://github.com/Kubenew/TurboMemory/discussions)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kubenew/TurboMemory&type=Date)](https://star-history.com/#Kubenew/TurboMemory&Date)
