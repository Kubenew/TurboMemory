import os, time
import numpy as np
from tqdm import tqdm

from .config import Config
from .utils.ids import sha1_id
from .utils.paths import ensure_dir
from .utils.hash import compute_phash, hamming_distance

from .schema_detect import detect_schema
from .entity_extract import extract_entities
from .memory.contradiction_graph import contradiction_score
from .memory.topics import assign_topic, update_centroid

from .embed.text_embedder import TextEmbedder
from .embed.clip_embedder import CLIPEmbedder
from .embed.batch import batched

from .pack4bit import pack_4bit
from .storage.sqlite_store import SQLiteStore
from .index.faiss_index import IVFIndex
from .index.merge_rank import merge_max
from .search import unpack_vectors_for_search
from .storage.tm_export import export_tm
from .storage.zip_export import export_zip

from .processors.text_processor import process_text_file
from .processors.pdf_processor import process_pdf
from .processors.image_processor import process_image
from .processors.video_processor import process_video

class AutoStructurerV5:
    def __init__(self, db_path="memory.sqlite", use_gpu=True):
        self.db_path = db_path
        self.use_gpu = use_gpu
        self.store = SQLiteStore(db_path)

        self.text_embed = TextEmbedder(Config.TEXT_EMBED_MODEL, use_gpu=use_gpu)
        self.clip_embed = CLIPEmbedder(Config.CLIP_MODEL, use_gpu=use_gpu)

        self.text_index_path = db_path + ".faiss_text.index"
        self.clip_index_path = db_path + ".faiss_clip.index"

        self.text_index = IVFIndex(dim=384, path=self.text_index_path, use_gpu=use_gpu)
        self.clip_index = IVFIndex(dim=512, path=self.clip_index_path, use_gpu=use_gpu)

        self._recent_hashes = []

    def _dedup(self, ref_path: str, max_dist=6):
        if not ref_path:
            return None
        ph = compute_phash(ref_path)
        for old in self._recent_hashes[-300:]:
            if hamming_distance(ph, old) <= max_dist:
                return None
        self._recent_hashes.append(ph)
        return ph

    def _process(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".txt",".md",".log",".html"]:
            return process_text_file(path)
        if ext == ".pdf":
            return process_pdf(path)
        if ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
            return process_image(path)
        if ext in [".mp4",".avi",".mov",".mkv"]:
            return process_video(path)
        raise ValueError("Unsupported file type: " + ext)

    def ingest_file(self, path: str):
        doc_id = sha1_id(os.path.abspath(path))
        chunks = self._process(path)
        if not chunks:
            return 0

        schema = detect_schema(" ".join([c["text"] for c in chunks[:5]]))

        all_texts = [c["text"] for c in chunks]
        text_vecs = self.text_embed.embed(all_texts)

        centroids = self.store.get_centroids()

        inserted = 0
        clip_jobs = []

        for i, ch in enumerate(chunks):
            cid = sha1_id(doc_id + str(i) + ch["text"][:200])

            ph = None
            if ch.get("ref_path"):
                ph = self._dedup(ch["ref_path"])
                if ph is None:
                    continue

            topic_id = 0
            if centroids is None:
                topic_id = 0
                self.store.upsert_centroid(0, text_vecs[i])
                centroids = self.store.get_centroids()
            else:
                topic_id, old = assign_topic(text_vecs[i], centroids)
                new_cent = update_centroid(old, text_vecs[i], alpha=0.05)
                self.store.upsert_centroid(topic_id, new_cent)
                centroids = self.store.get_centroids()

            rec = {
                "chunk_id": cid,
                "doc_id": doc_id,
                "source": ch.get("source","unknown"),
                "schema": schema,
                "t_start": ch.get("t_start",0.0),
                "t_end": ch.get("t_end",0.0),
                "text": ch["text"],
                "entities": extract_entities(ch["text"]),
                "topic": topic_id,
                "contradiction": contradiction_score(ch["text"]),
                "confidence": 0.85 if len(ch["text"])>30 else 0.55,
                "ref_path": ch.get("ref_path"),
                "phash": ph,
                "created_at": time.time()
            }
            self.store.insert_chunk(rec)

            packed, scale, zero = pack_4bit(text_vecs[i])
            vid = self.store.insert_vector(cid, "text", text_vecs.shape[1], packed, scale, zero)

            inserted += 1

            if ch.get("ref_path"):
                clip_jobs.append((cid, ch["ref_path"]))

        text_rows = self.store.fetch_vectors("text")
        ids = np.array([r[0] for r in text_rows], dtype=np.int64)
        vecs = unpack_vectors_for_search(text_rows)
        self.text_index.add(vecs, ids)
        self.text_index.save()

        if clip_jobs:
            img_paths = [p for _, p in clip_jobs]
            clip_vecs = self.clip_embed.embed_images(img_paths)

            clip_ids = []
            for (cid, _), v in zip(clip_jobs, clip_vecs):
                packed, scale, zero = pack_4bit(v)
                vid = self.store.insert_vector(cid, "clip", v.shape[0], packed, scale, zero)
                clip_ids.append(vid)

            clip_rows = self.store.fetch_vectors("clip")
            ids2 = np.array([r[0] for r in clip_rows], dtype=np.int64)
            vecs2 = unpack_vectors_for_search(clip_rows)
            self.clip_index.add(vecs2, ids2)
            self.clip_index.save()

        return inserted

    def search(self, query: str, mode="hybrid", top_k=10):
        results_text=[]
        results_clip=[]

        if mode in ["text","hybrid"]:
            q = self.text_embed.embed([query])
            D,I = self.text_index.search(q, top_k=top_k)
            for score, vid in zip(D[0], I[0]):
                if vid < 0:
                    continue
                for r in self.store.fetch_vectors("text"):
                    if r[0] == int(vid):
                        results_text.append((r[1], float(score), "text"))
                        break

        if mode in ["clip","hybrid"]:
            q = self.clip_embed.embed_texts([query])
            D,I = self.clip_index.search(q, top_k=top_k)
            for score, vid in zip(D[0], I[0]):
                if vid < 0:
                    continue
                for r in self.store.fetch_vectors("clip"):
                    if r[0] == int(vid):
                        results_clip.append((r[1], float(score), "clip"))
                        break

        merged = merge_max(results_text, results_clip, top_k=top_k)

        out=[]
        for cid, score, via in merged:
            row = self.store.fetch_chunk(cid)
            if not row:
                continue
            (chunk_id, doc_id, source, schema, t_start, t_end, text,
             entities_json, topic, contradiction, confidence, ref_path, phash, created_at) = row
            out.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": source,
                "schema": schema,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "text": text,
                "topic": f"topic_{topic}",
                "contradiction": float(contradiction),
                "confidence": float(confidence),
                "ref_path": ref_path,
                "score": float(score),
                "via": via
            })
        return out

    def export_tm(self, out_tm: str, zip_path=None):
        export_dir = export_tm(self.db_path, self.text_index_path, self.clip_index_path, out_tm)
        if zip_path:
            return export_zip(export_dir, zip_path)
        return out_tm