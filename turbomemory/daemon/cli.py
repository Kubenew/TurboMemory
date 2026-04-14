#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def pid_path(root: str) -> Path:
    return Path(root) / 'lock' / 'consolidator.pid'


def log_path(root: str) -> Path:
    return Path(root) / 'lock' / 'consolidator.log'


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_daemon(root: str, interval_sec: int, log_level: str = 'INFO') -> None:
    pfile = pid_path(root)
    pfile.parent.mkdir(parents=True, exist_ok=True)

    if pfile.exists():
        try:
            pid = int(pfile.read_text().strip())
            if is_running(pid):
                print(f'Daemon already running (pid={pid})')
                return
        except (ValueError, OSError):
            pass

    log_file = log_path(root)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        '-m', 'turbomemory.daemon.consolidator',
        '--root', root,
        '--daemon',
        '--interval_sec', str(interval_sec),
        '--log-level', log_level,
    ]

    with open(log_file, 'a') as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=lf,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            start_new_session=True,
        )

    pfile.write_text(str(proc.pid), encoding='utf-8')
    print(f'Daemon started pid={proc.pid}, log: {log_file}')


def stop_daemon(root: str) -> None:
    pfile = pid_path(root)
    if not pfile.exists():
        print('No PID file found.')
        return

    pid = int(pfile.read_text().strip())
    try:
        os.kill(pid, 15)
        for _ in range(10):
            if not is_running(pid):
                break
            time.sleep(0.3)
        else:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    except OSError as e:
        logger.warning(f'Stop error: {e}')

    try:
        pfile.unlink()
    except OSError:
        pass

    print('Daemon stopped.')


def status_daemon(root: str) -> None:
    pfile = pid_path(root)
    if not pfile.exists():
        print('Daemon not running.')
        return

    pid = int(pfile.read_text().strip())
    if is_running(pid):
        print(f'Daemon running pid={pid}')
    else:
        print('PID file exists but process is dead.')
        try:
            pfile.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description='TurboMemory daemon manager')
    parser.add_argument('--root', default='turbomemory_data')
    parser.add_argument('cmd', choices=['start', 'stop', 'status'])
    parser.add_argument('--interval_sec', type=int, default=120)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    if args.cmd == 'start':
        start_daemon(args.root, args.interval_sec, args.log_level)
    elif args.cmd == 'stop':
        stop_daemon(args.root)
    else:
        status_daemon(args.root)


if __name__ == '__main__':
    main()