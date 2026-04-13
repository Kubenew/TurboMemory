"""Test basic add and search in TurboMemory v3."""

import os
import shutil
from turbomemory import TurboMemoryV3
from turbomemory.config import TurboMemoryConfig


def test_add_search():
    root = "./test_store_v3"
    if os.path.exists(root):
        shutil.rmtree(root)
    
    tm = TurboMemoryV3(root=root, config=TurboMemoryConfig())
    
    # Add memories
    tm.add(text="TurboMemory stores embeddings in SQLite.", topic="arch")
    tm.add(text="FAISS IVF-PQ accelerates vector search.", topic="faiss")
    tm.add(text="WAL provides durability for writes.", topic="wal")
    
    # Search
    results = tm.search(query="What is used for storage?", top_k=5)
    
    assert len(results) > 0, "Search should return results"
    assert results[0]["text"], "Each result should have text"
    
    print(f"✓ Added 3 memories, found {len(results)} results")
    print(f"Top result: [{results[0]['score']:.3f}] {results[0]['text'][:60]}...")
    
    # Stats
    stats = tm.stats()
    print(f"✓ Stats: {stats}")
    
    tm.close()


if __name__ == "__main__":
    test_add_search()