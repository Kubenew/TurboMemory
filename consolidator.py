import os
import json
import argparse
import numpy as np
from datetime import datetime
from turbomemory import TurboMemory, dequantize_vector, quantize_vector, cosine_sim


def consolidate_topic(
    tm: TurboMemory,
    topic: str,
    similarity_threshold: float = 0.92,
    min_entropy: float = 0.12,
    max_chunks: int = 200
):
    topic_data = tm.load_topic(topic)
    chunks = topic_data.get("chunks", [])

    if not chunks:
        return {"topic": topic, "before": 0, "after": 0, "removed": 0}

    for c in chunks:
        c["_emb"] = dequantize_vector(c["embedding_q"])

    chunks.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    kept = []
    removed = 0

    for c in chunks:
        if c.get("entropy", 0.0) < min_entropy:
            removed += 1
            continue

        duplicate = False
        for k in kept:
            sim = cosine_sim(c["_emb"], k["_emb"])
            if sim >= similarity_threshold:
                duplicate = True
                removed += 1
                break

        if not duplicate:
            kept.append(c)

        if len(kept) >= max_chunks:
            break

    for c in kept:
        c.pop("_emb", None)

    all_embs = [dequantize_vector(c["embedding_q"]) for c in kept]
    centroid = np.mean(np.vstack(all_embs), axis=0)

    topic_data["chunks"] = kept
    topic_data["centroid_q"] = quantize_vector(centroid, bits=8)
    topic_data["updated"] = datetime.utcnow().isoformat() + "Z"

    tm.save_topic(topic_data)

    return {"topic": topic, "before": len(chunks), "after": len(kept), "removed": removed}


def rewrite_memory_md(tm: TurboMemory):
    header = ["# TurboMemory Index (v0.1)\n"]
    lines = header[:]

    for fn in os.listdir(tm.topics_dir):
        if not fn.endswith(".tmem"):
            continue

        path = os.path.join(tm.topics_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            topic_data = json.load(f)

        topic = topic_data["topic"]
        chunks = topic_data.get("chunks", [])

        if not chunks:
            continue

        chunks.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        best = chunks[0]

        entry_id = fn.replace(".tmem", "")[:5]
        score = float(best.get("confidence", 0.5))
        summary = best.get("text", "")[:150]
        file_ref = os.path.relpath(path, tm.root)

        line = f"{entry_id} | {topic} | {summary} | {score:.2f} | {datetime.utcnow().date()} | {file_ref}\n"
        lines.append(line)

    tm.write_index_lines(lines)


def main():
    parser = argparse.ArgumentParser(description="TurboMemory Consolidator v0.1")
    parser.add_argument("--root", default="turbomemory_data", help="TurboMemory data directory")
    parser.add_argument("--threshold", type=float, default=0.92, help="dedupe similarity threshold")
    parser.add_argument("--min_entropy", type=float, default=0.12, help="minimum entropy threshold")
    parser.add_argument("--max_chunks", type=int, default=200, help="max chunks per topic")
    args = parser.parse_args()

    tm = TurboMemory(root=args.root)

    topics = []
    for fn in os.listdir(tm.topics_dir):
        if fn.endswith(".tmem"):
            path = os.path.join(tm.topics_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                topics.append(json.load(f)["topic"])

    if not topics:
        print("No topics found.")
        return

    print(f"Consolidating {len(topics)} topics...")

    total_removed = 0
    for t in topics:
        res = consolidate_topic(
            tm,
            t,
            similarity_threshold=args.threshold,
            min_entropy=args.min_entropy,
            max_chunks=args.max_chunks
        )
        total_removed += res["removed"]
        print(f"[{t}] {res['before']} -> {res['after']} (removed {res['removed']})")

    rewrite_memory_md(tm)

    print(f"Done. Removed total: {total_removed}")
    print("MEMORY.md rewritten.")


if __name__ == "__main__":
    main()
