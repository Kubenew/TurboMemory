import argparse
import time
import os
import shutil
from turbomemory.autostructurer.pipeline import AutoStructurerV5

def main():
    p = argparse.ArgumentParser("AutoStructurer-v5 daemon")
    p.add_argument("--watch", required=True)
    p.add_argument("--db", default="memory.sqlite")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--poll", type=float, default=2.0)

    args = p.parse_args()

    os.makedirs(args.watch, exist_ok=True)
    done_dir = os.path.join(args.watch, "_done")
    err_dir = os.path.join(args.watch, "_error")
    os.makedirs(done_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)

    a = AutoStructurerV5(db_path=args.db, use_gpu=args.gpu)

    print("Watching:", args.watch)
    while True:
        files = [f for f in os.listdir(args.watch) if not f.startswith("_")]
        for f in files:
            path = os.path.join(args.watch, f)
            if not os.path.isfile(path):
                continue

            try:
                print("Ingesting:", path)
                a.ingest_file(path)
                shutil.move(path, os.path.join(done_dir, f))
            except Exception as e:
                print("ERROR ingesting:", path, "->", e)
                shutil.move(path, os.path.join(err_dir, f))

        time.sleep(args.poll)

if __name__ == "__main__":
    main()