# TurboMemory v0.2

TurboMemory is a layered memory system inspired by Claude memory design,
enhanced with TurboQuant-style packed quantization.

## Features (v0.2)
- SQLite index (`db/index.sqlite`)
- Packed quantization (4-bit / 6-bit / 8-bit)
- Topic centroid prefilter for fast retrieval
- Contradiction detection + confidence decay
- Background consolidation daemon (forked subprocess)

## Install
```bash
pip install -r requirements.txt
```

## Usage

### Add memory
```bash
python cli.py add_memory --topic turboquant.video --text "TurboQuant-v3 uses block matching for motion estimation." --bits 6
```

### Query
```bash
python cli.py query --query "How does TurboQuant handle video?" --k 5
```

### Stats
```bash
python cli.py stats
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
