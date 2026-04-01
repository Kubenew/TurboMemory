#!/usr/bin/env python3
# TurboMemory v0.2 consolidator
# - runs in background daemon mode
# - forked subprocess safe
# - dedupe + prune + rewrite MEMORY.md
# - updates SQLite index

import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from turbomemory import TurboMemory, dequantize_packed, quantize_packed, cosine_sim


def consolidate_topic(
    tm: TurboMemory,
    topic: str,
    similarity_threshold: float = 0.93,
    min_entropy: float = 0.10,
    staleness_prune: float = 0.90,
    max_chunks: int = 300
):
    topic_data = tm.load_topic(topic)
    chunks = topic_data.get("chunks", [])

    if not chunks:
        return {"topic": topic, "before": 0, "after": 0, "removed": 0}

    # decode embeddings
    for c in chunks:
        c["_emb"] = dequantize_packed(c["embedding_q"])

    # sort by confidence desc
    chunks.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    kept = []
    removed = 0

    for c in chunks:
        if c.get("entropy", 0.0) < min_entropy:
            removed += 1
            continue

        if c.get("staleness", 0.0) > staleness_prune:
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

    # recompute centroid
    all_embs = [dequantize_packed(c["embedding_q"]) for c in kept]
    centroid = np.mean(np.vstack(all_embs), axis=0)

    topic_data["chunks"] = kept
    topic_data["centroid_q"] = quantize_packed(centroid, bits=8)
    topic_data["updated"] = datetime.utcnow().isoformat() + "Z"

    tm.save_topic(topic_data)

    # update sqlite
    file_ref = os.path.relpath(tm._topic_path(topic), tm.root)
    tm._upsert_topic_to_db(topic_data, file_ref)
    for c in kept:
        tm._upsert_chunk_to_db(topic, c)

    return {"topic": topic, "before": len(chunks), "after": len(kept), "removed": removed}


def rewrite_memory_md(tm: TurboMemory):
    header = ["# TurboMemory Index (v0.2)\n"]
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


def run_once(tm: TurboMemory, args):
    topics = []
    for fn in os.listdir(tm.topics_dir):
        if fn.endswith(".tmem"):
            path = os.path.join(tm.topics_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                topics.append(json.load(f)["topic"])

    if not topics:
        print("No topics found.")
        return

    total_removed = 0
    for t in topics:
        res = consolidate_topic(
            tm,
            t,
            similarity_threshold=args.threshold,
            min_entropy=args.min_entropy,
            staleness_prune=args.staleness_prune,
            max_chunks=args.max_chunks
        )
        total_removed += res["removed"]
        print(f"[{t}] {res['before']} -> {res['after']} (removed {res['removed']})")

    rewrite_memory_md(tm)
    print(f"Done. Removed total: {total_removed}")
    print("MEMORY.md rewritten.")


def daemon_loop(tm: TurboMemory, args):
    print("TurboMemory consolidator daemon started.")
    print(f"interval_sec={args.interval_sec}")

    while True:
        try:
            run_once(tm, args)
        except Exception as e:
            print("Consolidation error:", str(e))

        time.sleep(args.interval_sec)


def main():
    parser = argparse.ArgumentParser(description="TurboMemory Consolidator v0.2")
    parser.add_argument("--root", default="turbomemory_data", help="TurboMemory data directory")
    parser.add_argument("--threshold", type=float, default=0.93, help="dedupe similarity threshold")
    parser.add_argument("--min_entropy", type=float, default=0.10, help="minimum entropy threshold")
    parser.add_argument("--staleness_prune", type=float, default=0.90, help="prune if staleness > threshold")
    parser.add_argument("--max_chunks", type=int, default=300, help="max chunks per topic")
    parser.add_argument("--daemon", action="store_true", help="run as background loop")
    parser.add_argument("--interval_sec", type=int, default=120, help="daemon sleep interval")
    args = parser.parse_args()

    tm = TurboMemory(root=args.root)

    if args.daemon:
        daemon_loop(tm, args)
    else:
        run_once(tm, args)


if __name__ == "__main__":
    main()
