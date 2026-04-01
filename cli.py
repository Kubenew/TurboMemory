#!/usr/bin/env python3
"""TurboMemory CLI v0.4 - Command-line interface for TurboMemory."""

import argparse
import json
import sys
from turbomemory.turbomemory import TurboMemory, TurboMemoryConfig


def cmd_add_turn(tm: TurboMemory, args):
    ref = tm.add_turn(args.role, args.text, session_file=args.session)
    print(f"Logged: {ref}")


def cmd_add_memory(tm: TurboMemory, args):
    source_ref = args.source
    if args.log:
        source_ref = tm.add_turn("user", args.text, session_file=args.session)

    chunk_id = tm.add_memory(
        topic=args.topic,
        text=args.text,
        confidence=args.confidence,
        bits=args.bits,
        source_ref=source_ref,
        ttl_days=args.ttl_days,
    )
    if chunk_id:
        print(f"Added memory to topic '{args.topic}' (chunk_id: {chunk_id})")
    else:
        print("Memory excluded by rules.")


def cmd_query(tm: TurboMemory, args):
    if args.verify:
        results = tm.verify_and_score(
            args.query,
            k=args.k,
            top_topics=args.top_topics,
            min_confidence=args.min_confidence,
        )
        for score, topic, chunk, verif in results:
            print("=" * 70)
            print(f"SCORE: {score:.4f}")
            print(f"TOPIC: {topic}")
            print(f"ID: {chunk.get('chunk_id')}")
            print(f"CONF: {chunk.get('confidence')}")
            print(f"STALE: {chunk.get('staleness')}")
            print(f"TEXT: {chunk.get('text')}")
            print(f"VERIFIED: {verif.verified} (score: {verif.verification_score:.3f})")
            print(f"CROSS-REFS: {verif.cross_references}")
            if verif.flags:
                print(f"FLAGS: {', '.join(verif.flags)}")
    else:
        hits = tm.query(
            args.query,
            k=args.k,
            top_topics=args.top_topics,
            min_confidence=args.min_confidence,
            require_verification=args.require_verified,
        )
        if not hits:
            print("No results.")
            return
        for score, topic, chunk in hits:
            print("=" * 70)
            print(f"SCORE: {score:.4f}")
            print(f"TOPIC: {topic}")
            print(f"ID: {chunk.get('chunk_id')}")
            print(f"CONF: {chunk.get('confidence')}")
            print(f"STALE: {chunk.get('staleness')}")
            print(f"QUALITY: {chunk.get('quality_score', 0):.3f}")
            print(f"VERIFIED: {chunk.get('verified', False)}")
            print(f"TEXT: {chunk.get('text')}")
            print(f"TS: {chunk.get('timestamp')}")


def cmd_stats(tm: TurboMemory, args):
    metrics = tm.get_metrics()
    print("=== TurboMemory Metrics ===")
    print(f"Topics: {metrics.total_topics}")
    print(f"Chunks: {metrics.total_chunks}")
    print(f"Avg Confidence: {metrics.avg_confidence:.3f}")
    print(f"Avg Staleness: {metrics.avg_staleness:.3f}")
    print(f"Avg Quality: {metrics.avg_quality:.3f}")
    print(f"Expired: {metrics.expired_chunks}")
    print(f"Contradicted: {metrics.contradicted_chunks}")
    print(f"Verified: {metrics.verified_chunks}")
    print(f"Storage: {metrics.storage_bytes / 1024:.1f} KB")
    print(f"Consolidation Runs: {metrics.consolidation_runs}")
    if metrics.last_consolidation:
        print(f"Last Consolidation: {metrics.last_consolidation}")
    print(f"Chunks Removed by Consolidation: {metrics.chunks_removed_by_consolidation}")
    if metrics.topic_health:
        print("\n=== Topic Health ===")
        for topic, health in sorted(metrics.topic_health.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(health * 20) + "░" * (20 - int(health * 20))
            print(f"  {topic:30s} [{bar}] {health:.2f}")


def cmd_quality(tm: TurboMemory, args):
    quality = tm.get_chunk_quality(args.topic, args.chunk_id)
    print(f"Quality Score: {quality.overall:.3f}")
    print(f"  Confidence: {quality.confidence_component:.3f}")
    print(f"  Freshness:  {quality.freshness_component:.3f}")
    print(f"  Specificity: {quality.specificity_component:.3f}")
    print(f"  Verification: {quality.verification_component:.3f}")
    if quality.flags:
        print(f"  Flags: {', '.join(quality.flags)}")


def cmd_decay_quality(tm: TurboMemory, args):
    count = tm.decay_quality()
    print(f"Applied quality decay to {count} chunks.")


def cmd_rebuild(tm: TurboMemory, args):
    tm.rebuild_index()
    print("SQLite index rebuilt from topic files.")


def cmd_expire_ttl(tm: TurboMemory, args):
    count = tm.expire_ttl()
    print(f"Expired {count} chunks.")


def cmd_backup(tm: TurboMemory, args):
    path = tm.backup(args.backup_path)
    print(f"Backup created at: {path}")


def cmd_restore(tm: TurboMemory, args):
    tm.restore(args.backup_path)
    print("Restore completed.")


def cmd_export(tm: TurboMemory, args):
    if args.topic:
        data = tm.export_topic(args.topic, include_embeddings=args.with_embeddings)
    else:
        data = tm.export_all(include_embeddings=args.with_embeddings)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_import(tm: TurboMemory, args):
    with open(args.file, "r") as f:
        items = json.load(f)
    result = tm.bulk_import(items)
    print(f"Imported: {result['imported']}, Excluded: {result['excluded']}, Failed: {result['failed']}")


def cmd_merge(tm: TurboMemory, args):
    count = tm.merge_topics(args.source, args.target)
    print(f"Merged {count} chunks from '{args.source}' into '{args.target}'.")


def cmd_metrics(tm: TurboMemory, args):
    metrics = tm.get_metrics()
    print(json.dumps(metrics.to_dict(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="TurboMemory CLI v0.4")
    parser.add_argument("--root", default="turbomemory_data", help="data directory")
    parser.add_argument("--config", default=None, help="config file path")

    sub = parser.add_subparsers(dest="cmd")

    # add_turn
    p = sub.add_parser("add_turn", help="append session log line")
    p.add_argument("--role", required=True, help="user/assistant/system")
    p.add_argument("--text", required=True)
    p.add_argument("--session", default=None)

    # add_memory
    p = sub.add_parser("add_memory", help="store memory chunk")
    p.add_argument("--topic", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--bits", type=int, default=6, choices=[4, 6, 8])
    p.add_argument("--source", default=None)
    p.add_argument("--log", action="store_true")
    p.add_argument("--session", default=None)
    p.add_argument("--ttl_days", type=float, default=None, help="TTL in days")

    # query
    p = sub.add_parser("query", help="search memory")
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--top_topics", type=int, default=5)
    p.add_argument("--min_confidence", type=float, default=0.0)
    p.add_argument("--verify", action="store_true", help="enable verification")
    p.add_argument("--require_verified", action="store_true", help="only return verified results")

    # stats
    sub.add_parser("stats", help="show stats with topic health")

    # quality
    p = sub.add_parser("quality", help="get quality score for a chunk")
    p.add_argument("--topic", required=True)
    p.add_argument("--chunk_id", required=True)

    # decay_quality
    sub.add_parser("decay_quality", help="apply quality decay to all chunks")

    # rebuild index
    sub.add_parser("rebuild", help="rebuild sqlite index from topic files")

    # expire TTL
    sub.add_parser("expire_ttl", help="remove expired chunks")

    # backup
    p = sub.add_parser("backup", help="create backup")
    p.add_argument("--backup_path", required=True, help="backup directory")

    # restore
    p = sub.add_parser("restore", help="restore from backup")
    p.add_argument("--backup_path", required=True, help="backup directory")

    # export
    p = sub.add_parser("export", help="export topics")
    p.add_argument("--topic", default=None, help="topic name (omit for all)")
    p.add_argument("--with_embeddings", action="store_true", help="include embeddings")

    # import
    p = sub.add_parser("import", help="bulk import from JSON file")
    p.add_argument("--file", required=True, help="JSON file path")

    # merge topics
    p = sub.add_parser("merge", help="merge source topic into target")
    p.add_argument("--source", required=True, help="source topic")
    p.add_argument("--target", required=True, help="target topic")

    # metrics (JSON)
    sub.add_parser("metrics", help="output metrics as JSON")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    config = None
    if args.config:
        config = TurboMemoryConfig.from_file(args.config)
    elif args.root:
        config = TurboMemoryConfig(root=args.root)

    tm = TurboMemory(config=config)

    try:
        if args.cmd == "add_turn":
            cmd_add_turn(tm, args)
        elif args.cmd == "add_memory":
            cmd_add_memory(tm, args)
        elif args.cmd == "query":
            cmd_query(tm, args)
        elif args.cmd == "stats":
            cmd_stats(tm, args)
        elif args.cmd == "quality":
            cmd_quality(tm, args)
        elif args.cmd == "decay_quality":
            cmd_decay_quality(tm, args)
        elif args.cmd == "rebuild":
            cmd_rebuild(tm, args)
        elif args.cmd == "expire_ttl":
            cmd_expire_ttl(tm, args)
        elif args.cmd == "backup":
            cmd_backup(tm, args)
        elif args.cmd == "restore":
            cmd_restore(tm, args)
        elif args.cmd == "export":
            cmd_export(tm, args)
        elif args.cmd == "import":
            cmd_import(tm, args)
        elif args.cmd == "merge":
            cmd_merge(tm, args)
        elif args.cmd == "metrics":
            cmd_metrics(tm, args)
        else:
            print("Unknown command.")
    finally:
        tm.close()


if __name__ == "__main__":
    main()
