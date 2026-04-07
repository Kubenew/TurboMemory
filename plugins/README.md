# TurboMemory Plugins

Plugins extend TurboMemory with specialized functionality.

## AutoStructurer

AI-powered ETL pipeline for unstructured data (text, audio, video, images):
- Whisper transcript extraction (GPU)
- OCR (EasyOCR GPU)  
- Scene keyframe detection
- Batch embedding (SentenceTransformer + CLIP)
- FAISS GPU IVF-PQ index
- Topic clustering
- Export to TurboMemory TM format

### Installation

```bash
pip install turbomemory[autostructurer]
```

### Usage

```python
from turbomemory.plugins import AutoStructurerV5

# Ingest files
a = AutoStructurerV5(db_path="memory.sqlite", use_gpu=True)
a.ingest_file("video.mp4")

# Search
results = a.search("query", mode="hybrid", top_k=10)
```

### CLI

```bash
# Ingest file
python -m plugins.autostructurer.cli ingest video.mp4 --db memory.sqlite --gpu

# Search  
python -m plugins.autostructurer.cli search "query" --db memory.sqlite --mode hybrid

# Daemon mode
python -m plugins.autostructurer.daemon --watch inbox --db memory.sqlite --gpu
```

## Writing Custom Plugins

Plugins can use the TurboMemory storage interface:

```python
from turbomemory import TurboMemoryWriter, TurboMemorySearch, ChunkMetadata
import numpy as np

# Write
writer = TurboMemoryWriter("data/")
writer.add(
    text="Your text",
    vector=np.array([...]),  # embedding
    metadata=ChunkMetadata(
        chunk_id="chunk_1",
        source="my_plugin",
        schema="custom"
    )
)

# Search
search = TurboMemorySearch("data/")
results = search.search_hybrid(text_query, clip_query, top_k=10)
```