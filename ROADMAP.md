# TurboMemory Roadmap 🚀

> Claude-style long-term memory with 4/6/8-bit TurboQuant compression — runs on a laptop.

---

# Vision

TurboMemory aims to become:
- **SQLite-like simplicity** for semantic memory
- **Parquet-like storage efficiency** for embeddings via TurboQuant
- **Edge-first replication** for distributed memory networks
- **Self-healing memory** via background consolidation

---

# Current Status: v0.4 (✅ Implemented)

- [x] Core memory engine with SQLite + quantization
- [x] Packed 4-bit / 6-bit / 8-bit embedding storage
- [x] Topic-based segmentation with centroid prefilter
- [x] Retrieval verification with cross-checking
- [x] Quality scoring + decay
- [x] Exclusion rules (what NOT to store)
- [x] Self-healing consolidation (merge/prune, contradiction resolution)
- [x] Observability + metrics
- [x] Plugin system for extensibility
- [x] LangChain integrations (retriever, chat history)
- [x] CLI with stats, search, verification flags
- [x] Streamlit dashboard scaffold

**What's NOT implemented yet:** PyPI, benchmarks, storage format spec, hybrid search, server mode

---

# v0.5 — Storage Format Specification (TMF v1)

**Goal:** define a stable storage format (Parquet-like for semantic memory).

### Deliverables
- [ ] Define "TurboMemory Format (TMF)" v1 spec:
  - `.tmindex` (SQLite metadata index)
  - `.tmvec` (packed vectors file)
  - `.tmlog` (append-only transcript/event log)
  - `.tmmeta.json` (schema + model metadata)
- [ ] Schema versioning + migrations
- [ ] Checksum verification + corruption detection
- [ ] Deterministic serialization format for vectors
- [ ] Fast export/import tooling

### Metrics
- Storage can be copied between machines and loaded instantly

---

# v0.6 — Hybrid Search (Vector + Keyword)

**Goal:** become usable for real RAG and enterprise search.

### Deliverables
- [ ] BM25 or keyword search fallback
- [ ] Scoring fusion:
  - vector similarity score
  - keyword score
  - recency score
  - confidence score
- [ ] Metadata filters:
  - time range
  - topic filter
  - tags/namespace filter
- [ ] Query explain output (debug mode)

### Metrics
- Improved retrieval accuracy on real datasets
- Stable query API supporting filters

---

# v0.7 — Server Mode (Microservice API)

**Goal:** allow TurboMemory to run as a service.

### Deliverables
- [ ] REST API (FastAPI):
  - `/add`
  - `/bulk_add`
  - `/query`
  - `/delete`
  - `/stats`
- [ ] Multi-tenant namespaces
- [ ] API key auth
- [ ] Docker image
- [ ] Rate limiting / request validation

### Metrics
- Stable service running on VPS with <500MB RAM

---

# v0.8 — Replication & Sync (Edge-first)

**Goal:** local-first semantic DB that syncs like Git/WAL.

### Deliverables
- [ ] Log-based replication:
  - Node A asks Node B for missing offsets
  - Node B streams missing `.tmlog` events
- [ ] Conflict handling policy:
  - Append-only ID uniqueness
  - Merge rules for duplicates
- [ ] Sync over HTTP
- [ ] Optional encryption for replication payload

### Metrics
- 2 nodes can sync 100k memories reliably
- Idempotent sync (safe to retry)

---

# v0.9 — Index Acceleration (Optional HNSW/IVF)

**Goal:** make retrieval fast for millions of vectors.

### Deliverables
- [ ] Optional HNSW index module
- [ ] IVF centroid bucket acceleration
- [ ] Caching layer for hot topics
- [ ] Multi-thread query execution
- [ ] Vector block prefetching

### Metrics
- Query latency remains low at 1M+ chunks on CPU

---

# v1.0 — Production Core Release

**Goal:** stable foundation usable in real apps.

### Deliverables
- [ ] Stable API freeze
- [ ] Full documentation site
- [ ] Full test coverage for packing/indexing
- [ ] Verified storage integrity guarantees
- [ ] Stable migrations
- [ ] Release notes + changelog discipline

### Metrics
- Safe upgrades
- Reproducible performance

---

# v1.1+ — Distributed Sharding / Cluster Mode

**Goal:** scale horizontally across machines.

### Deliverables
- [ ] Sharding by:
  - topic
  - time range
  - centroid hash
- [ ] Router node (fan-out query)
- [ ] Distributed top-k merge
- [ ] Node health metrics + monitoring endpoints

---

# v2.0 — Semantic Data Lake Index

**Goal:** semantic indexing layer for S3/object storage.

### Deliverables
- [ ] S3-compatible backend for `.tmvec` and `.tmlog`
- [ ] Scalable metadata store option (Postgres)
- [ ] Caching proxy nodes
- [ ] Batch ingestion pipelines
- [ ] Integration examples with LlamaIndex/LangChain

---

# Contributor-Friendly Areas

If you want to contribute, these are high-impact modules:

### Beginner (Good First Issues)
- Docs improvements
- Tests
- CLI UX
- Examples and demos

### Intermediate
- Benchmark harness
- Scoring fusion logic
- SQLite optimization

### Advanced
- Packed embedding codec improvements
- Replication protocol
- HNSW/IVF index integration
- Corruption recovery tools

---

# Guiding Principles

TurboMemory is built around:
- **Local-first**
- **Small footprint**
- **Append-only logs**
- **Cheap compressed storage**
- **Self-healing consolidation**
- **Portable file format**
- **No cloud dependency**

---

# Long-Term Goal

TurboMemory should become:
> "The default open storage format for semantic memory and compressed embeddings."

If you want to help build that future, join the project 🚀
