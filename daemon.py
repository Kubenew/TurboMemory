#!/usr/bin/env python3
# TurboMemory v0.2 daemon manager
# Starts consolidator.py in a forked subprocess and writes PID file.

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def pid_path(root: str):
    return Path(root) / "lock" / "consolidator.pid"


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except:
        return False


def start_daemon(root: str, interval_sec: int):
    pfile = pid_path(root)
    pfile.parent.mkdir(parents=True, exist_ok=True)

    if pfile.exists():
        try:
            pid = int(pfile.read_text().strip())
            if is_running(pid):
                print(f"Daemon already running (pid={pid})")
                return
        except:
            pass

    cmd = [
        sys.executable,
        "consolidator.py",
        "--root", root,
        "--daemon",
        "--interval_sec", str(interval_sec)
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        start_new_session=True
    )

    pfile.write_text(str(proc.pid), encoding="utf-8")
    print(f"Daemon started pid={proc.pid}")


def stop_daemon(root: str):
    pfile = pid_path(root)
    if not pfile.exists():
        print("No PID file found.")
        return

    pid = int(pfile.read_text().strip())
    try:
        os.kill(pid, 15)
        time.sleep(0.3)
    except Exception as e:
        print("Stop error:", str(e))

    try:
        pfile.unlink()
    except:
        pass

    print("Daemon stopped.")


def status_daemon(root: str):
    pfile = pid_path(root)
    if not pfile.exists():
        print("Daemon not running.")
        return

    pid = int(pfile.read_text().strip())
    if is_running(pid):
        print(f"Daemon running pid={pid}")
    else:
        print("PID file exists but process is dead.")


def main():
    parser = argparse.ArgumentParser(description="TurboMemory daemon manager")
    parser.add_argument("--root", default="turbomemory_data")
    parser.add_argument("cmd", choices=["start", "stop", "status"])
    parser.add_argument("--interval_sec", type=int, default=120)
    args = parser.parse_args()

    if args.cmd == "start":
        start_daemon(args.root, args.interval_sec)
    elif args.cmd == "stop":
        stop_daemon(args.root)
    else:
        status_daemon(args.root)


if __name__ == "__main__":
    main()
