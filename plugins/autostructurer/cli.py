import argparse
from plugins.autostructurer.pipeline import AutoStructurerV5

def main():
    p = argparse.ArgumentParser("AutoStructurer-v5")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("ingest")
    pi.add_argument("path")
    pi.add_argument("--db", default="memory.sqlite")
    pi.add_argument("--gpu", action="store_true")

    ps = sub.add_parser("search")
    ps.add_argument("query")
    ps.add_argument("--db", default="memory.sqlite")
    ps.add_argument("--top-k", type=int, default=10)
    ps.add_argument("--mode", choices=["text", "clip", "hybrid"], default="hybrid")

    pe = sub.add_parser("export")
    pe.add_argument("--db", default="memory.sqlite")
    pe.add_argument("--out", default="memory.tm")
    pe.add_argument("--zip", default=None)

    args = p.parse_args()

    if args.cmd == "ingest":
        a = AutoStructurerV5(db_path=args.db, use_gpu=args.gpu)
        n = a.ingest_file(args.path)
        print("Inserted chunks:", n)

    elif args.cmd == "search":
        a = AutoStructurerV5(db_path=args.db, use_gpu=True)
        results = a.search(args.query, mode=args.mode, top_k=args.top_k)
        for r in results:
            print(f"\n[{r['score']:.3f}] via={r['via']} schema={r['schema']} source={r['source']} topic={r['topic']}")
            print(f"time: {r['t_start']:.1f}-{r['t_end']:.1f} conf={r['confidence']:.2f} contradiction={r['contradiction']:.2f}")
            if r.get("ref_path"):
                print("ref:", r["ref_path"])
            print(r["text"][:400])

    elif args.cmd == "export":
        a = AutoStructurerV5(db_path=args.db, use_gpu=True)
        out = a.export_tm(args.out, zip_path=args.zip)
        print("Exported:", out)

    else:
        p.print_help()

if __name__ == "__main__":
    main()