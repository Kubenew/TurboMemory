# TurboMemory ⚡

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

## Use Cases

TurboMemory can be used as:
- compressed semantic archive for large text corpora
- long-term memory backend for AI agents
- offline semantic search engine for notes/transcripts
- lightweight RAG store on laptop/VPS/edge devices
- persistent audit log + recall layer for autonomous systems

---

## Quickstart

### Install

```bash
git clone https://github.com/Kubenew/TurboMemory.git
cd TurboMemory
pip install -r requirements.txt
```

### CLI Usage

```bash
python cli.py add_memory --topic notes --text "TurboMemory stores semantic chunks efficiently."
python cli.py query --query "semantic storage"
```

### Python Usage

```python
from turbomemory import TurboMemory

tm = TurboMemory(root="./tm_data")

tm.add_memory(
    topic="notes",
    text="TurboMemory stores semantic chunks efficiently.",
    ttl_days=365
)

results = tm.query("semantic storage", k=5)

for score, topic, chunk in results:
    print(score, chunk["text"])
```

---

## Repository Structure

```
TurboMemory/
├── turbomemory/              # core package
│   ├── __init__.py
│   ├── turbomemory.py        # core storage/retrieval engine
│   ├── plugins/              # plugin system
│   └── integrations/         # LangChain, etc.
├── cli.py                    # command-line interface
├── consolidator.py           # merging/pruning logic
├── daemon.py                 # background maintenance daemon
├── dashboard.py              # Streamlit dashboard
├── tests/                    # unit tests
├── benchmarks/               # benchmark scripts
├── examples/                 # usage examples
└── notebooks/                # Colab demos
```

---

## Architecture Overview

TurboMemory uses a layered approach:

### 1) Index Layer (SQLite)

SQLite stores:
- chunk metadata
- topic pointers
- timestamps
- quality/confidence scores
- optional verification flags

### 2) Vector Storage Layer (Packed Embeddings)

Embeddings are stored in compact binary form:
- 8-bit packed
- 6-bit packed
- 4-bit packed

This enables high compression ratios compared to float32 vectors.

### 3) Transcript Log Layer (Append-only)

Raw events are stored as append-only logs:
- insert operations
- merges/consolidation events
- pruning actions

This design supports replication/sync in future versions.

### 4) Consolidation Daemon

A background process periodically:
- merges similar chunks
- removes duplicates
- decays outdated memory confidence
- rebuilds centroids/index if needed

---

## Design Principles

TurboMemory is built around:
- **local-first operation**
- **small footprint**
- **portable storage format**
- **append-only durability**
- **compression-first vector storage**
- **self-healing maintenance**
- **minimal dependencies**

---

## Roadmap

See [ROADMAP.md](ROADMAP.md)

Upcoming major milestones:
- v0.3: stability + CI + packaging
- v0.4: benchmarks + profiling
- v0.5: TurboMemory Format (TMF v1) stable file spec
- v0.6: hybrid search (keyword + vector fusion)
- v0.7: server mode (FastAPI)
- v0.8: replication / edge sync

---

## Benchmarks (Planned)

Benchmarks will include:
- insert throughput (chunks/sec)
- query latency (top-k)
- disk usage by bit-width (float32 vs 8-bit vs 6-bit vs 4-bit)
- recall quality comparisons

---

## Contributing

Contributions are welcome.

Start here:
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Good First Issues](https://github.com/Kubenew/TurboMemory/labels/good%20first%20issue)

We actively label issues as:
- `good first issue`
- `help wanted`
- `advanced`

---

## License

MIT License.

---

## Disclaimer

TurboMemory is an experimental project.

Interfaces and storage formats may change until v1.0.

---

## Contact

Open an issue or discussion for feedback, feature requests, or collaboration.
