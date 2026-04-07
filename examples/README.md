# TurboMemory Examples

Example scripts demonstrating TurboMemory usage.

## Basic Examples

### 1. Quick Start
```python
# examples/quickstart.py
from turbomemory import TurboMemory

tm = TurboMemory(root="./data")
tm.add_memory("notes", "My first note")
results = tm.query("first note")
for score, topic, chunk in results:
    print(f"{score}: {chunk['text']}")
```

### 2. CLI Usage
```bash
# Add memory
python cli.py add_memory --topic notes --text "Hello World"

# Query
python cli.py query --query "Hello" --k 5

# Stats
python cli.py stats
```

## Integration Examples

### LangChain RAG
```python
# examples/langchain_rag.py
from turbomemory.integrations import get_turbo_memory_vectorstore

vectorstore = get_turbo_memory_vectorstore(root="./data", topic="docs")
vectorstore.add_texts(["doc1 content", "doc2 content"])
docs = vectorstore.similarity_search("query")
```

### LlamaIndex
```python
# examples/llamaindex_example.py
from turbomemory.integrations import getTurboMemoryIndex

index = getTurboMemoryIndex(root="./data")
query_engine = index.as_query_engine()
response = query_engine.query("What is TurboMemory?")
```

## Advanced Examples

### Custom Consolidation
```python
# examples/consolidation.py
from turbomemory import TurboMemory
from turbomemory.consolidator import run_consolidation

tm = TurboMemory(root="./data")

# Run consolidation
result = run_consolidation(tm, topic="notes")
print(f"Merged {result['merged']} chunks")
```

### Hybrid Search
```python
# examples/hybrid_search.py
from turbomemory.hybrid_search import HybridSearchEngine

engine = HybridSearchEngine(root="./data")
results = engine.search("search query", top_k=10)
```

### Replication Sync
```python
# examples/replication.py
from turbomemory.replication import create_sync

sync = create_sync("./data", "http://remote:8000")
result = sync.sync()
print(f"Synced with {result['pull']['imported']} events")
```

## Running Examples

```bash
# Install dependencies
pip install -e ".[all]"

# Run quickstart
python examples/quickstart.py

# Run CLI
python cli.py add_memory --topic test --text "Example"
```

## Example Datasets

The `examples/` folder includes sample scripts for:
- Adding sample documents
- Bulk importing from JSON
- Exporting topics
- Running queries with filters

## Need Help?

- Check the [main documentation](../README.md)
- Open an [issue](https://github.com/Kubenew/TurboMemory/issues)
- Join [discussions](https://github.com/Kubenew/TurboMemory/discussions)
