import sqlite3, json, time
import numpy as np

class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path=db_path
        self._init()

    def _init(self):
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            source TEXT,
            schema TEXT,
            t_start REAL,
            t_end REAL,
            text TEXT,
            entities_json TEXT,
            topic INTEGER,
            contradiction REAL,
            confidence REAL,
            ref_path TEXT,
            phash TEXT,
            created_at REAL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS vectors(
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT,
            kind TEXT,
            dim INTEGER,
            packed BLOB,
            scale REAL,
            zero REAL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS topic_centroids(
            topic INTEGER PRIMARY KEY,
            dim INTEGER,
            vec BLOB
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS contradictions(
            a TEXT,
            b TEXT,
            score REAL,
            PRIMARY KEY(a,b)
        )
        """)
        conn.commit(); conn.close()

    def insert_chunk(self, rec: dict):
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("""INSERT OR REPLACE INTO chunks(
            chunk_id, doc_id, source, schema, t_start, t_end, text,
            entities_json, topic, contradiction, confidence, ref_path, phash, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            rec["chunk_id"], rec["doc_id"], rec["source"], rec["schema"],
            float(rec.get("t_start",0.0)), float(rec.get("t_end",0.0)),
            rec["text"], json.dumps(rec["entities"], ensure_ascii=False),
            int(rec["topic"]), float(rec["contradiction"]), float(rec["confidence"]),
            rec.get("ref_path"), rec.get("phash"), float(rec.get("created_at", time.time()))
        ))
        conn.commit(); conn.close()

    def insert_vector(self, chunk_id, kind, dim, packed, scale, zero):
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("""INSERT INTO vectors(chunk_id, kind, dim, packed, scale, zero)
                       VALUES (?,?,?,?,?,?)""", (chunk_id, kind, int(dim), packed, float(scale), float(zero)))
        vid=cur.lastrowid
        conn.commit(); conn.close()
        return vid

    def fetch_vectors(self, kind):
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("SELECT vector_id, chunk_id, dim, packed, scale, zero FROM vectors WHERE kind=?", (kind,))
        rows=cur.fetchall()
        conn.close()
        return rows

    def fetch_chunk(self, chunk_id):
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("""SELECT chunk_id, doc_id, source, schema, t_start, t_end, text,
                                entities_json, topic, contradiction, confidence, ref_path, phash, created_at
                         FROM chunks WHERE chunk_id=?""", (chunk_id,))
        row=cur.fetchone()
        conn.close()
        return row

    def get_centroids(self):
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("SELECT topic, dim, vec FROM topic_centroids ORDER BY topic ASC")
        rows=cur.fetchall()
        conn.close()
        if not rows:
            return None
        dim=rows[0][1]
        cent=np.vstack([np.frombuffer(r[2], dtype=np.float32)[:dim] for r in rows])
        return cent

    def upsert_centroid(self, topic: int, vec: np.ndarray):
        vec = vec.astype(np.float32)
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("""INSERT OR REPLACE INTO topic_centroids(topic, dim, vec) VALUES (?,?,?)""",
                    (int(topic), int(vec.shape[0]), vec.tobytes()))
        conn.commit(); conn.close()

    def insert_contradiction(self, a, b, score):
        conn=sqlite3.connect(self.db_path)
        cur=conn.cursor()
        cur.execute("INSERT OR REPLACE INTO contradictions(a,b,score) VALUES (?,?,?)", (a,b,float(score)))
        conn.commit(); conn.close()