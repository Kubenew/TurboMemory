#!/usr/bin/env python3
import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    from turbomemory import TurboMemory
    
    parser = argparse.ArgumentParser(description='TurboMemory Consolidator')
    parser.add_argument('--root', default='turbomemory_data', help='TurboMemory data directory')
    parser.add_argument('--threshold', type=float, default=0.93, help='dedupe similarity threshold')
    parser.add_argument('--min_entropy', type=float, default=0.10, help='minimum entropy threshold')
    parser.add_argument('--staleness_prune', type=float, default=0.90, help='prune if staleness > threshold')
    parser.add_argument('--max_chunks', type=int, default=300, help='max chunks per topic')
    parser.add_argument('--merge_threshold', type=float, default=0.85, help='merge similarity threshold')
    parser.add_argument('--no_make_absolute', action='store_true', help='disable vague-to-absolute conversion')
    parser.add_argument('--daemon', action='store_true', help='run as background loop')
    parser.add_argument('--interval_sec', type=int, default=120, help='daemon sleep interval')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    args.convert_to_absolute = not args.no_make_absolute
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    from .consolidator import run_once, daemon_loop
    
    tm = TurboMemory(root=args.root)
    try:
        if args.daemon:
            daemon_loop(tm, args)
        else:
            run_once(tm, args)
    finally:
        tm.close()


if __name__ == '__main__':
    main()