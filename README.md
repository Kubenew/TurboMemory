# TurboMemory v0.4

TurboMemory is a layered memory system inspired by Claude Code's memory architecture,
enhanced with TurboQuant-style packed quantization, retrieval verification, and self-healing consolidation.

## Design Principles

> **Memory = index, not storage**
> `MEMORY.md` is always loaded but stores only pointers (~150 chars/line). Actual knowledge lives in topic files, fetched on demand.

> **3-layer bandwidth-aware design**
> - Layer 1: Index (always loaded)
> - Layer 2: Topic files (on-demand)
> - Layer 3: Transcripts (never read, only appended)

> **Strict write discipline**
> Write to file, then update index. Never dump content into the index. Prevents entropy and context pollution.

> **Background memory rewriting (autoDream)**
> Merges duplicates, resolves contradictions, converts vague language to absolute statements, and aggressively prunes. Memory is continuously edited, not just appended.

> **Staleness is first-class**
> If memory ≠ reality, memory is wrong. Code-derived facts are never stored. The index is forcibly truncated.

> **Retrieval is skeptical, not blind**
> Memory is a hint, not truth. The model must verify before using.

> **What we don't store is the real insight**
> No debugging logs, no code structure, no PR history. If it's derivable, don't persist it.

## Features (v0.4)

### Core
- SQLite index with connection pooling (`db/index.sqlite`)
- Packed quantization (4-bit / 6-bit / 8-bit)
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
- Blocks: debug output, code snippets, secrets, file paths, PR history
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
- Storage tracking

### Operations
- Backup and restore
- Bulk import/export (JSON)
- Topic merging and splitting
- Background consolidation daemon with logging

## Install

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Usage

### Add memory

```bash
python cli.py add_memory --topic turboquant.video --text "TurboQuant-v3 uses block matching for motion estimation." --bits 6
```

### Add memory with TTL (expires in 7 days)

```bash
python cli.py add_memory --topic news --text "Breaking news..." --ttl_days 7
```

### Query

```bash
# Basic query
python cli.py query --query "How does TurboQuant handle video?" --k 5

# With verification
python cli.py query --query "TurboQuant video" --verify

# Only verified results
python cli.py query --query "TurboQuant" --require_verified
```

### Stats with topic health

```bash
python cli.py stats
```

Output:
```
=== TurboMemory Metrics ===
Topics: 5
Chunks: 42
Avg Confidence: 0.780
Avg Staleness: 0.120
Avg Quality: 0.650
...

=== Topic Health ===
  turboquant.video               [████████████████░░░░] 0.82
  python.tips                    [██████████████░░░░░░] 0.70
```

### Get chunk quality

```bash
python cli.py quality --topic turboquant.video --chunk_id c0001
```

### Consolidate once

```bash
python consolidator.py
```

### Run consolidator daemon

```bash
python daemon.py start --root turbomemory_data --interval_sec 120
python daemon.py status --root turbomemory_data
python daemon.py stop --root turbomemory_data
```

### Rebuild SQLite index (repair)

```bash
python cli.py rebuild
```

### Expire TTL chunks

```bash
python cli.py expire_ttl
```

### Backup / Restore

```bash
python cli.py backup --backup_path ./backup_2024
python cli.py restore --backup_path ./backup_2024
```

### Export / Import

```bash
# Export all topics
python cli.py export

# Export single topic with embeddings
python cli.py export --topic my_topic --with_embeddings

# Bulk import from JSON
python cli.py import --file memories.json
```

### Merge topics

```bash
python cli.py merge --source old_topic --target new_topic
```

### Metrics (JSON)

```bash
python cli.py metrics
```

## Configuration

Create a `config.json` file:

```json
{
  "root": "turbomemory_data",
  "model_name": "all-MiniLM-L6-v2",
  "default_bits": 6,
  "pool_size": 5,
  "default_ttl_days": null,
  "max_chunks_per_topic": 300,
  "contradiction_decay": 0.6,
  "min_confidence": 0.0,
  "enable_verification": true,
  "verification_threshold": 0.7,
  "min_cross_refs": 2,
  "enable_exclusions": true,
  "min_quality_threshold": 0.3,
  "quality_decay_rate": 0.01
}
```

Use it with:

```bash
python cli.py --config config.json add_memory --topic test --text "Hello"
```

## Programmatic Usage

```python
from turbomemory import TurboMemory

with TurboMemory(root="my_memory") as tm:
    # Add memory (returns None if excluded)
    chunk_id = tm.add_memory("python", "Python is a programming language")

    # Query with verification
    results = tm.verify_and_score("programming language", k=5)
    for score, topic, chunk, verif in results:
        status = "VERIFIED" if verif.verified else "UNVERIFIED"
        print(f"{status} [{topic}]: {chunk['text']}")

    # Get metrics
    metrics = tm.get_metrics()
    print(f"Health: {metrics.avg_quality:.2f}")
```

## Architecture

```
MEMORY.md (index, always loaded)
    ↓
topics/*.tmem (structured topic files, loaded on demand)
    ↓
sessions/*.jsonl (immutable logs, appended only)
    ↓
db/index.sqlite (fast retrieval, connection pooled)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
