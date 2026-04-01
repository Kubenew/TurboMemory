import argparse
from turbomemory import TurboMemory


def cmd_add_turn(tm: TurboMemory, args):
    ref = tm.add_turn(args.role, args.text, session_file=args.session)
    print(f"Logged: {ref}")


def cmd_add_memory(tm: TurboMemory, args):
    source_ref = args.source
    if args.log:
        source_ref = tm.add_turn("user", args.text, session_file=args.session)

    tm.add_memory(
        topic=args.topic,
        text=args.text,
        confidence=args.confidence,
        bits=args.bits,
        source_ref=source_ref
    )
    print(f"Added memory to topic '{args.topic}'")


def cmd_query(tm: TurboMemory, args):
    hits = tm.query(args.query, k=args.k, topic=args.topic)

    if not hits:
        print("No results.")
        return

    for sim, topic, chunk in hits:
        print("=" * 70)
        print(f"SIM: {sim:.4f}")
        print(f"TOPIC: {topic}")
        print(f"ID: {chunk.get('chunk_id')}")
        print(f"CONF: {chunk.get('confidence')}")
        print(f"TEXT: {chunk.get('text')}")
        print(f"SOURCES: {chunk.get('source_refs')}")


def cmd_stats(tm: TurboMemory, args):
    print(tm.stats())


def main():
    parser = argparse.ArgumentParser(description="TurboMemory CLI v0.1")
    parser.add_argument("--root", default="turbomemory_data", help="data directory")

    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("add_turn", help="append session log line")
    p.add_argument("--role", required=True, help="user/assistant/system")
    p.add_argument("--text", required=True)
    p.add_argument("--session", default=None, help="session file override")

    p = sub.add_parser("add_memory", help="store memory chunk")
    p.add_argument("--topic", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--bits", type=int, default=6)
    p.add_argument("--source", default=None, help="optional source ref")
    p.add_argument("--log", action="store_true", help="also log as user turn")
    p.add_argument("--session", default=None, help="session file override")

    p = sub.add_parser("query", help="search memory")
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--topic", default=None)

    sub.add_parser("stats", help="show stats")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    tm = TurboMemory(root=args.root)

    if args.cmd == "add_turn":
        cmd_add_turn(tm, args)
    elif args.cmd == "add_memory":
        cmd_add_memory(tm, args)
    elif args.cmd == "query":
        cmd_query(tm, args)
    elif args.cmd == "stats":
        cmd_stats(tm, args)
    else:
        print("Unknown command.")


if __name__ == "__main__":
    main()
