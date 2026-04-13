PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid            TEXT UNIQUE,
    text            TEXT NOT NULL,

    topic           TEXT,
    source          TEXT,
    agent_id        TEXT,

    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,

    confidence      REAL DEFAULT 0.5,
    importance     REAL DEFAULT 0.5,

    ttl_seconds     INTEGER DEFAULT NULL,

    access_count    INTEGER DEFAULT 0,
    last_accessed   INTEGER DEFAULT 0,

    hash            TEXT,
    extra_json      TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic);
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id       INTEGER PRIMARY KEY,
    dim             INTEGER NOT NULL,
    dtype           TEXT NOT NULL,
    blob            BLOB NOT NULL,

    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tags (
    memory_id       INTEGER NOT NULL,
    tag             TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_mem ON tags(memory_id);

CREATE TABLE IF NOT EXISTS edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    id_a            INTEGER NOT NULL,
    id_b            INTEGER NOT NULL,
    relation_type   TEXT NOT NULL,
    score           REAL NOT NULL,
    created_at      INTEGER NOT NULL,

    FOREIGN KEY(id_a) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(id_b) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_a ON edges(id_a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(id_b);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type);

-- FTS5 for BM25-style keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(text, content='memories', content_rowid='id');

-- WAL tracking
CREATE TABLE IF NOT EXISTS wal_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              INTEGER NOT NULL,
    op              TEXT NOT NULL,
    memory_id       INTEGER,
    details         TEXT
);

-- Topic centroids for fast prefiltering
CREATE TABLE IF NOT EXISTS topic_centroids (
    topic               TEXT PRIMARY KEY,
    centroid_dim       INTEGER NOT NULL,
    centroid_dtype     TEXT NOT NULL,
    centroid_blob      BLOB NOT NULL,
    updated_at         INTEGER NOT NULL
);