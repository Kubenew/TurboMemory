"""Test WAL append and replay."""

import os
import shutil
from turbomemory.storage.wal import WAL


def test_wal_append_replay():
    root = "./test_wal"
    if os.path.exists(root):
        shutil.rmtree(root)
    
    # Create WAL
    wal = WAL(root)
    
    # Append records
    wal.append({"op": "add", "id": 1, "text": "hello", "topic": "test"})
    wal.append({"op": "add", "id": 2, "text": "world", "topic": "test"})
    wal.append({"op": "reinforce", "memory_id": 1, "delta": 0.1})
    
    # Replay
    out = []
    wal.replay(lambda r: out.append(r))
    
    assert len(out) == 3, f"Expected 3 records, got {len(out)}"
    assert out[0]["op"] == "add"
    assert out[1]["text"] == "world"
    assert out[2]["op"] == "reinforce"
    
    print(f"✓ WAL replayed {len(out)} records")
    
    wal.close()


if __name__ == "__main__":
    test_wal_append_replay()