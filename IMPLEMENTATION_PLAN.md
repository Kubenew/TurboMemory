# TurboMemory Implementation Plan

This document outlines the implementation plan for reaching v1.0 production release and beyond.

---

## v0.5 — Storage Format Specification (TMF v1)
**Target: Q2 2026 | Effort: 3-4 weeks**

### Goals
Define a stable, portable storage format that can be copied between machines.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v0.5.1 | Define TMF v1 schema (`.tmindex`, `.tmvec`, `.tmlog`, `.tmmeta.json`) | 1 week | None |
| v0.5.2 | Implement TMF file I/O layer | 1 week | v0.5.1 |
| v0.5.3 | Add schema versioning + migration logic | 3 days | v0.5.2 |
| v0.5.4 | Implement checksum verification + corruption detection | 3 days | v0.5.2 |
| v0.5.5 | Add deterministic serialization for vectors | 2 days | v0.5.1 |
| v0.5.6 | Build export/import CLI tools | 3 days | v0.5.2, v0.5.3 |
| v0.5.7 | Write TMF spec documentation | 2 days | v0.5.1 |

### Files to Modify/Create
- `turbomemory/formats/tmf.py` — TMF file format handler
- `turbomemory/formats/tmf_index.py` — `.tmindex` operations
- `turbomemory/formats/tmf_vector.py` — `.tmvec` operations
- `turbomemory/formats/tmf_log.py` — `.tmlog` operations
- `turbomemory/formats/migrations.py` — schema migration

---

## v0.6 — Hybrid Search (Vector + Keyword)
**Target: Q2 2026 | Effort: 4-5 weeks**

### Goals
Support real RAG and enterprise search with keyword + vector fusion.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v0.6.1 | Implement BM25 keyword search fallback | 1 week | None |
| v0.6.2 | Build scoring fusion pipeline | 1 week | v0.6.1 |
| v0.6.3 | Add metadata filters (time range, topic, tags, namespace) | 1 week | v0.6.2 |
| v0.6.4 | Implement query explain output (debug mode) | 3 days | v0.6.2 |
| v0.6.5 | Add configurable scoring weights via config | 2 days | v0.6.2 |
| v0.6.6 | Integration tests for hybrid search | 1 week | v0.6.3 |

### Files to Modify/Create
- `turbomemory/search/keyword.py` — BM25 implementation
- `turbomemory/search/fusion.py` — scoring fusion
- `turbomemory/search/filters.py` — metadata filters
- `turbomemory/search/explain.py` — query explanation

---

## v0.7 — Server Mode (Microservice API)
**Target: Q3 2026 | Effort: 4-5 weeks**

### Goals
Allow TurboMemory to run as a REST API service.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v0.7.1 | Set up FastAPI project structure | 3 days | None |
| v0.7.2 | Implement `/add` endpoint | 3 days | v0.7.1 |
| v0.7.3 | Implement `/bulk_add` endpoint | 3 days | v0.7.2 |
| v0.7.4 | Implement `/query` endpoint | 3 days | v0.7.2 |
| v0.7.5 | Implement `/delete` endpoint | 2 days | v0.7.1 |
| v0.7.6 | Implement `/stats` endpoint | 2 days | v0.7.1 |
| v0.7.7 | Add multi-tenant namespaces | 1 week | v0.7.5 |
| v0.7.8 | Add API key authentication | 3 days | v0.7.7 |
| v0.7.9 | Create Docker image | 3 days | v0.7.8 |
| v0.7.10 | Add rate limiting + request validation | 3 days | v0.7.9 |

### Files to Modify/Create
- `server/main.py` — FastAPI app
- `server/routes/add.py` — add endpoints
- `server/routes/query.py` — query endpoints
- `server/routes/stats.py` — stats endpoints
- `server/auth.py` — API key auth
- `server/tenants.py` — multi-tenant logic
- `Dockerfile`
- `docker-compose.yml`

---

## v0.8 — Replication & Sync (Edge-first)
**Target: Q3 2026 | Effort: 5-6 weeks**

### Goals
Enable local-first semantic DB that syncs like Git/WAL.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v0.8.1 | Design replication protocol (request/stream) | 1 week | None |
| v0.8.2 | Implement `.tmlog` event streaming | 1 week | v0.8.1 |
| v0.8.3 | Add conflict resolution (append-only ID uniqueness) | 1 week | v0.8.2 |
| v0.8.4 | Implement duplicate merge rules | 1 week | v0.8.3 |
| v0.8.5 | Add HTTP sync protocol | 1 week | v0.8.4 |
| v0.8.6 | Add optional payload encryption | 3 days | v0.8.5 |
| v0.8.7 | Build sync CLI tools (sync push, sync pull, sync status) | 1 week | v0.8.5 |
| v0.8.8 | Integration tests for 2-node sync | 1 week | v0.8.7 |

### Files to Modify/Create
- `turbomemory/sync/protocol.py` — replication protocol
- `turbomemory/sync/event_log.py` — event streaming
- `turbomemory/sync/conflict.py` — conflict resolution
- `turbomemory/sync/http.py` — HTTP sync
- `turbomemory/sync/crypto.py` — encryption
- `cli.py` — add sync commands

---

## v0.9 — Index Acceleration (Optional HNSW/IVF)
**Target: Q3-Q4 2026 | Effort: 4-5 weeks**

### Goals
Make retrieval fast for millions of vectors on CPU.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v0.9.1 | Create optional HNSW index module | 1 week | None |
| v0.9.2 | Implement IVF centroid bucket acceleration | 1 week | v0.9.1 |
| v0.9.3 | Add caching layer for hot topics | 1 week | v0.9.2 |
| v0.9.4 | Implement multi-thread query execution | 3 days | v0.9.3 |
| v0.9.5 | Add vector block prefetching | 3 days | v0.9.4 |
| v0.9.6 | Make HNSW/IVF optional (feature flag) | 2 days | v0.9.5 |

### Files to Modify/Create
- `turbomemory/index/hnsw.py` — HNSW implementation
- `turbomemory/index/ivf.py` — IVF implementation
- `turbomemory/index/cache.py` — caching layer
- `turbomemory/index/query.py` — parallel query execution

---

## v1.0 — Production Core Release
**Target: Q4 2026 | Effort: 3-4 weeks**

### Goals
Stable foundation usable in real apps.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v1.0.1 | Freeze stable API surface | 1 week | All v0.x |
| v1.0.2 | Build documentation site (mkdocs) | 1 week | v1.0.1 |
| v1.0.3 | Achieve 90%+ test coverage | 1 week | v1.0.1 |
| v1.0.4 | Add storage integrity verification | 3 days | v0.5.x |
| v1.0.5 | Implement stable migrations | 1 week | v0.5.x |
| v1.0.6 | Write release notes + changelog | 2 days | v1.0.5 |
| v1.0.7 | Final security audit | 3 days | v1.0.6 |

### Files to Modify/Create
- `docs/` — mkdocs site
- `CHANGELOG.md`
- ` SECURITY.md`

---

## v1.1+ — Distributed Sharding / Cluster Mode
**Target: 2027 | Effort: 8-12 weeks**

### Goals
Scale horizontally across machines.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v1.1.1 | Implement sharding by topic | 2 weeks | v1.0 |
| v1.1.2 | Implement sharding by time range | 1 week | v1.1.1 |
| v1.1.3 | Implement sharding by centroid hash | 1 week | v1.1.2 |
| v1.1.4 | Build router node (fan-out query) | 2 weeks | v1.1.3 |
| v1.1.5 | Implement distributed top-k merge | 2 weeks | v1.1.4 |
| v1.1.6 | Add node health metrics + monitoring | 2 weeks | v1.1.5 |

---

## v2.0 — Semantic Data Lake Index
**Target: 2027 | Effort: 10-16 weeks**

### Goals
Semantic indexing layer for S3/object storage.

### Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| v2.0.1 | Implement S3-compatible backend for `.tmvec` | 3 weeks | v1.0 |
| v2.0.2 | Implement S3-compatible backend for `.tmlog` | 2 weeks | v2.0.1 |
| v2.0.3 | Add Postgres metadata store option | 2 weeks | v2.0.2 |
| v2.0.4 | Build caching proxy nodes | 2 weeks | v2.0.3 |
| v2.0.5 | Implement batch ingestion pipelines | 2 weeks | v2.0.4 |
| v2.0.6 | Add LlamaIndex integration example | 1 week | v2.0.5 |
| v2.0.7 | Add LangChain integration example | 1 week | v2.0.6 |

---

## Contributor Assignment Guide

### By Skill Level

**Beginner (Good First Issues)**
- v0.5.7: Write TMF spec documentation
- v0.6.4: Query explain output
- v0.6.5: Configurable scoring weights
- v1.0.6: Write release notes + changelog
- v2.0.6/7: Integration examples

**Intermediate**
- v0.5.2: TMF file I/O
- v0.5.3: Schema migrations
- v0.6.1: BM25 keyword search
- v0.6.2: Scoring fusion
- v0.7.1-10: Server mode
- v0.9.x: Index acceleration

**Advanced**
- v0.8.x: Replication protocol
- v0.9.x: HNSW implementation
- v1.1.x: Distributed sharding
- v2.0.x: S3 backend

---

## Timeline Summary

| Version | Target | Focus | Key Deliverables |
|---------|--------|-------|------------------|
| v0.5 | Q2 2026 | Storage Format | TMF v1 spec, export/import |
| v0.6 | Q2 2026 | Hybrid Search | BM25 + vector fusion, filters |
| v0.7 | Q3 2026 | Server Mode | FastAPI, Docker, multi-tenant |
| v0.8 | Q3 2026 | Replication | Edge sync, conflict resolution |
| v0.9 | Q3-Q4 2026 | Index Acceleration | HNSW/IVF, caching |
| v1.0 | Q4 2026 | Production | API freeze, docs, tests |
| v1.1 | 2027 | Sharding | Horizontal scaling |
| v2.0 | 2027 | Data Lake | S3 backend |

---

## Milestone Checkpoints

Each version should pass:
1. **CI pipeline** — all tests pass
2. **Type checking** — mypy clean
3. **Lint check** — ruff + black pass
4. **Benchmarks** — performance documented
5. **Documentation** — updated docs
6. **Changelog** — release notes written
