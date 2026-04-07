# TurboMemory Server

FastAPI-based server for TurboMemory with multi-tenant support.

## Quick Start

```bash
# Run the server
python -m turbomemory.api.server

# Or use the CLI
turbomemory-server --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t turbomemory .
docker run -p 8000:8000 turbomemory
```

## Environment Variables

- `TURBOMEMORY_ROOT` - Data directory (default: turbomemory_data)
- `TURBOMEMORY_MODEL` - Embedding model (default: all-MiniLM-L6-v2)
- `TURBOMEMORY_HOST` - Server host (default: 0.0.0.0)
- `TURBOMEMORY_PORT` - Server port (default: 8000)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/memory` | Add memory chunk |
| POST | `/memory/bulk` | Bulk import |
| GET | `/memory/{topic}` | Get topic |
| DELETE | `/memory/{topic}/{chunk_id}` | Delete chunk |
| POST | `/query` | Search memories |
| GET | `/topics` | List topics |
| GET | `/metrics` | System metrics |
| GET | `/stats` | Quick stats |
| GET | `/export` | Export data |
| POST | `/backup` | Create backup |
| POST | `/restore` | Restore backup |
| POST | `/sync` | Sync with remote |

## Multi-Tenant Mode

Enable namespace isolation:

```python
from turbomemory.api.server import ServerConfig, create_app

config = ServerConfig(
    enable_namespaces=True,
    root="./data"
)
app = create_app(config)
```

Access specific namespaces using the `Authorization` header:

```bash
curl -H "Authorization: namespace:user1" http://localhost:8000/query \
  -d '{"query": "search"}'
```

## Development

```bash
# Run with auto-reload
turbomemory-server --reload
```
