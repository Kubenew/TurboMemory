#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

import numpy as np

logger = logging.getLogger(__name__)


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class RLVisualizer:
    def __init__(self, output_dir: str = 'rl_visualizations'):
        self.output_dir = output_dir
        if MATPLOTLIB_AVAILABLE:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    def plot_memory_growth(self, memory_history: List[Dict], save_path: Optional[str] = None):
        if not MATPLOTLIB_AVAILABLE:
            logger.warning('Matplotlib not available')
            return

        timestamps = [m.get('timestamp') for m in memory_history]
        chunk_counts = [m.get('chunk_count', 0) for m in memory_history]

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(timestamps)), chunk_counts, marker='o')
        plt.title('Memory Growth Over Time')
        plt.xlabel('Step')
        plt.ylabel('Total Chunks')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f'Saved plot to {save_path}')
        plt.close()

    def plot_topic_distribution(self, topics: Dict[str, int], save_path: Optional[str] = None):
        if not MATPLOTLIB_AVAILABLE:
            logger.warning('Matplotlib not available')
            return

        if not topics:
            return

        plt.figure(figsize=(12, 6))
        plt.bar(topics.keys(), topics.values())
        plt.title('Topic Distribution')
        plt.xlabel('Topic')
        plt.ylabel('Chunk Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f'Saved plot to {save_path}')
        plt.close()

    def plot_confidence_distribution(self, chunks: List[Dict], save_path: Optional[str] = None):
        if not MATPLOTLIB_AVAILABLE:
            logger.warning('Matplotlib not available')
            return

        confidences = [c.get('confidence', 0.5) for c in chunks]

        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, edgecolor='black')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f'Saved plot to {save_path}')
        plt.close()

    def plot_quality_trends(self, quality_history: List[Dict], save_path: Optional[str] = None):
        if not MATPLOTLIB_AVAILABLE or not quality_history:
            logger.warning('Matplotlib not available or no data')
            return

        steps = [q.get('step') for q in quality_history]
        avg_quality = [q.get('avg_quality', 0) for q in quality_history]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, avg_quality, marker='o')
        plt.title('Quality Trends Over Time')
        plt.xlabel('Step')
        plt.ylabel('Average Quality')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f'Saved plot to {save_path}')
        plt.close()


class MemoryAnalyzer:
    def __init__(self, root: str):
        self.root = root

    def get_topic_stats(self) -> Dict[str, Any]:
        stats = {
            'total_topics': 0,
            'total_chunks': 0,
            'topic_sizes': {},
            'avg_confidence': 0,
        }

        topics_dir = os.path.join(self.root, 'topics')
        if not os.path.exists(topics_dir):
            return stats

        all_confidences = []
        for fn in os.listdir(topics_dir):
            if not fn.endswith('.tmem'):
                continue

            path = os.path.join(topics_dir, fn)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    chunks = data.get('chunks', [])
                    topic = data.get('topic', fn)

                    stats['total_topics'] += 1
                    stats['total_chunks'] += len(chunks)
                    stats['topic_sizes'][topic] = len(chunks)

                    all_confidences.extend([c.get('confidence', 0.5) for c in chunks])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f'Error reading {fn}: {e}')

        if all_confidences:
            stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)

        return stats

    def analyze_retention(self) -> Dict[str, Any]:
        retention = {
            'high_retention_topics': [],
            'low_retention_topics': [],
            'stale_chunks': 0,
        }

        topics_dir = os.path.join(self.root, 'topics')
        if not os.path.exists(topics_dir):
            return retention

        for fn in os.listdir(topics_dir):
            if not fn.endswith('.tmem'):
                continue

            path = os.path.join(topics_dir, fn)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    chunks = data.get('chunks', [])
                    topic = data.get('topic', fn)

                    stale_count = sum(1 for c in chunks if c.get('staleness', 0) > 0.7)
                    retention['stale_chunks'] += stale_count

                    if len(chunks) > 0:
                        retention_rate = (len(chunks) - stale_count) / len(chunks)
                        if retention_rate > 0.8:
                            retention['high_retention_topics'].append(topic)
                        elif retention_rate < 0.5:
                            retention['low_retention_topics'].append(topic)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f'Error reading {fn}: {e}')

        return retention


def generate_report(root: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    analyzer = MemoryAnalyzer(root)
    visualizer = RLVisualizer()

    report = {
        'topic_stats': analyzer.get_topic_stats(),
        'retention': analyzer.analyze_retention(),
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f'Report saved to {output_path}')

    return report