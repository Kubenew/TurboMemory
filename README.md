# TurboMemory

> **Claude-style long-term memory with 4/6/8-bit TurboQuant compression — runs on a laptop.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Kubenew/TurboMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/Kubenew/TurboMemory/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/turbomemory.svg)](https://badge.fury.io/py/turbomemory)
[![Downloads](https://pepy.tech/badge/turbomemory)](https://pepy.tech/project/turbomemory)

[🚀 Live Demo (Colab)](https://colab.research.google.com/github/Kubenew/TurboMemory/blob/main/notebooks/demo.ipynb) · [📖 Docs](#readme) · [🗺️ Roadmap](ROADMAP.md) · [💬 Discussions](https://github.com/Kubenew/TurboMemory/discussions)

---

## Why TurboMemory?

| Feature | TurboMemory | Mem0 | Zep | LangMem |
|---------|:-----------:|:----:|:---:|:-------:|
| **Embedding Compression** | ✅ 4/6/8-bit packed | ❌ | ❌ | ❌ |
| **Self-Healing (autoDream)** | ✅ Merge, dedupe, resolve contradictions | Partial | Partial | ❌ |
| **Retrieval Verification** | ✅ Cross-reference scoring | ❌ | ❌ | ❌ |
| **Quality Scoring** | ✅ Confidence + freshness + specificity | ❌ | ❌ | ❌ |
| **Exclusion Rules** | ✅ Configurable "what NOT to store" | ❌ | ❌ | ❌ |
| **Runs on Laptop** | ✅ SQLite + local models | ⚠️ Needs vector DB | ❌ Needs server | ⚠️ Needs vector DB |
| **Memory Size (10K chunks)** | ~5 MB (6-bit) | ~150 MB | ~200 MB | ~150 MB |
| **Open Source** | ✅ MIT | ✅ Apache 2.0 | ✅ | ✅ |
| **Plugin System** | ✅ Scorers, providers, storage | ❌ | ❌ | ❌ |

**The compression advantage:** TurboMemory's 6-bit quantization stores embeddings at **~25% the size** of full float32 with >0.95 cosine similarity. That means 10,000 memories in ~5 MB instead of ~150 MB.

---

## Want to Help? Pick a Good First Issue!

We especially need:
- **Benchmarks** — Compare vs Mem0, Zep, LangMem
- **LangChain integration** — Retriever + chat history (started, needs testing)
- **Web dashboard** — Streamlit app for browsing memories
- **Documentation** — Tutorials, API docs, architecture diagrams

👉 [Good First Issues](https://github.com/Kubenew/TurboMemory/labels/good%20first%20issue) · [Contributing Guide](CONTRIBUTING.md)

---

## Quick Start

```bash
pip install -e .
```

```python
from turbomemory import TurboMemory

with TurboMemory(root="my_memory") as tm:
    # Add memory
    tm.add_memory("python", "Python uses dynamic typing and garbage collection")

    # Query with verification
    results = tm.verify_and_score("How does Python work?")
    for score, topic, chunk, verif in results:
        print(f"{'✓' if verif.verified else '?'} {chunk['text']}")
```

---

## Design Principles

> **Memory = index, not storage**
> `MEMORY.md` stores only pointers (~150 chars/line). Actual knowledge lives in topic files.

> **3-layer bandwidth-aware design**
> Index (always) → Topics (on-demand) → Transcripts (append-only)

> **Strict write discipline**
> Write to file, then update index. Never dump content into the index.

> **Background memory rewriting (autoDream)**
> Merges duplicates, resolves contradictions, converts vague → absolute. Memory is continuously edited.

> **Staleness is first-class**
> If memory ≠ reality, memory is wrong. Code-derived facts are never stored.

> **Retrieval is skeptical, not blind**
> Memory is a hint, not truth. Cross-reference verification before use.

> **What we don't store is the real insight**
> No debug logs, no code structure, no PR history. If derivable, don't persist.

---

## Features

### Core
- SQLite index with connection pooling
- Packed quantization (4-bit / 6-bit / 8-bit) — **up to 8x compression**
- Topic centroid prefilter for fast retrieval
- Contradiction detection + confidence decay
- TTL (time-to-live) for memory chunks

### Verification
- Cross-reference verification across topics
- Agreement scoring between related chunks
- Contradiction flagging during retrieval
- Optional "verified-only" query mode

### Quality Scoring
- Per-chunk quality scores (confidence + freshness + specificity + verification)
- Automatic quality decay over time
- Quality-based ranking adjustments

### Exclusion Rules
- Configurable patterns for what NOT to store
- Blocks: debug output, code snippets, secrets, PR history
- Exclusion logging for auditability

### Self-Healing Consolidation
- Semantic merging of similar chunks
- Contradiction resolution (older chunks decayed)
- Vague-to-absolute language conversion
- Aggressive deduplication and pruning

### Observability
- Per-topic health scores (0.0 - 1.0)
- Consolidation event logging
- Comprehensive metrics (JSON output)

### Plugin System
- Custom quality scorers
- Custom embedding providers
- Custom storage backends (Redis, PostgreSQL, etc.)
- Custom verification strategies

### Integrations
- **LangChain** — `TurboMemoryRetriever`, `TurboMemoryChatMessageHistory`
- **CrewAI** — Memory provider example
- More coming: AutoGen, LlamaIndex, Haystack

---

## Usage

### CLI

```bash
# Add memory
python cli.py add_memory --topic turboquant.video --text "TurboQuant-v3 uses block matching" --bits 6

# Query with verification
python cli.py query --query "How does TurboQuant work?" --verify

# Stats with topic health
python cli.py stats

# Consolidate
python consolidator.py
```

### LangChain

```python
from turbomemory.integrations.langchain import TurboMemoryRetriever

retriever = TurboMemoryRetriever(root="my_memory", k=5, enable_verification=True)
docs = retriever.invoke("What is TurboQuant?")
```

### Streamlit Dashboard

```bash
pip install streamlit
streamlit run dashboard.py
```

---

## Compression Benchmarks

| Bits | Original (384-dim) | Compressed | Ratio | Similarity |
|------|-------------------|------------|-------|------------|
| 4-bit | 1536 bytes | ~192 bytes | 8.0x | >0.90 |
| 6-bit | 1536 bytes | ~288 bytes | 5.3x | >0.95 |
| 8-bit | 1536 bytes | ~384 bytes | 4.0x | >0.99 |

Run benchmarks: `python -m benchmarks.compression_bench`

---

## Architecture

```
MEMORY.md (index, always loaded)
    ↓
topics/*.tmem (structured topic files, loaded on demand)
    ↓
sessions/*.jsonl (immutable logs, appended only)
    ↓
db/index.sqlite (fast retrieval, connection pooled)
    ↓
plugins/ (custom scorers, providers, storage, verification)
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Good first issues:** [View issues](https://github.com/Kubenew/TurboMemory/labels/good%20first%20issue)

**Roadmap:** [ROADMAP.md](ROADMAP.md)

---

## License

MIT — see [LICENSE](LICENSE)
