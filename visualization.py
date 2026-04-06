#!/usr/bin/env python3
"""Visualization and analytics for RL + TurboMemory integration."""

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
    """Visualization tools for RL training with memory."""

    def __init__(self, output_dir: str = "rl_visualizations"):
        self.output_dir = output_dir
        if MATPLOTLIB_AVAILABLE:
            os.makedirs(output_dir, exist_ok=True)

    def plot_learning_curve(
        self,
        rewards: List[float],
        window_size: int = 10,
        title: str = "Learning Curve",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available")
            return None

        episodes = range(len(rewards))
        smoothed = self._smooth(rewards, window_size)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        ax.plot(episodes, smoothed, color='blue', linewidth=2, label=f'Smoothed (window={window_size})')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = save_path or os.path.join(self.output_dir, "learning_curve.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_memory_usage(
        self,
        memory_sizes: List[int],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(memory_sizes, color='green')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Memory Chunks')
        ax1.set_title('Memory Growth Over Time')
        ax1.grid(True, alpha=0.3)

        ax2.hist(memory_sizes[-100:], bins=20, color='green', alpha=0.7)
        ax2.set_xlabel('Memory Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Recent Memory Distribution')
        ax2.grid(True, alpha=0.3)

        path = save_path or os.path.join(self.output_dir, "memory_usage.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_subgoal_success_rate(
        self,
        subgoal_history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE or not subgoal_history:
            return None

        subgoals = {}
        for entry in subgoal_history:
            name = entry.get('label', 'unknown')
            if name not in subgoals:
                subgoals[name] = {'success': 0, 'total': 0}
            subgoals[name]['total'] += 1
            if entry.get('success', False):
                subgoals[name]['success'] += 1

        names = list(subgoals.keys())
        rates = [subgoals[n]['success'] / max(1, subgoals[n]['total']) for n in names]
        totals = [subgoals[n]['total'] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        bars = ax1.bar(names, rates, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Subgoal')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Subgoal Success Rates')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, rate in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.2f}', ha='center', va='bottom', fontsize=9)

        ax2.bar(names, totals, color='coral', alpha=0.8)
        ax2.set_xlabel('Subgoal')
        ax2.set_ylabel('Attempt Count')
        ax2.set_title('Subgoal Attempt Counts')
        ax2.grid(True, alpha=0.3, axis='y')

        path = save_path or os.path.join(self.output_dir, "subgoal_success.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_query_performance(
        self,
        query_times: List[float],
        query_types: List[str],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE or not query_times:
            return None

        unique_types = list(set(query_types))
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_types)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for i, qtype in enumerate(unique_types):
            times = [t for t, qt in zip(query_times, query_types) if qt == qtype]
            ax1.plot(times, alpha=0.6, label=qtype, color=colors[i])

        ax1.set_xlabel('Query Number')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Query Performance Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        avg_times = [np.mean([t for t, qt in zip(query_times, query_types) if qt == qtype])
                     for qtype in unique_types]
        ax2.bar(unique_types, avg_times, color=colors)
        ax2.set_xlabel('Query Type')
        ax2.set_ylabel('Average Time (seconds)')
        ax2.set_title('Average Query Time by Type')
        ax2.grid(True, alpha=0.3, axis='y')

        path = save_path or os.path.join(self.output_dir, "query_performance.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_reward_distribution(
        self,
        rewards: List[float],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(rewards, bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Reward Distribution')
        axes[0].grid(True, alpha=0.3)

        axes[1].boxplot(rewards, vert=True)
        axes[1].set_ylabel('Reward')
        axes[1].set_title('Reward Box Plot')
        axes[1].grid(True, alpha=0.3)

        percentiles = [10, 25, 50, 75, 90]
        pct_values = np.percentile(rewards, percentiles)
        axes[2].plot(percentiles, pct_values, 'o-', color='purple', markersize=8)
        axes[2].set_xlabel('Percentile')
        axes[2].set_ylabel('Reward')
        axes[2].set_title('Reward Percentiles')
        axes[2].grid(True, alpha=0.3)

        path = save_path or os.path.join(self.output_dir, "reward_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_episode_statistics(
        self,
        rewards: List[float],
        steps: List[int],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        smoothed = self._smooth(rewards, 10)
        axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
        axes[0, 0].plot(smoothed, label='Smoothed', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(rewards, steps, alpha=0.3, color='green')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].set_title('Reward vs Steps')
        axes[1, 0].grid(True, alpha=0.3)

        rolling_mean = self._rolling_mean(rewards, 50)
        rolling_std = self._rolling_std(rewards, 50)
        x = range(50, len(rewards) + 1)
        axes[1, 1].fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.3)
        axes[1, 1].plot(x, rolling_mean, linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title('Rolling Mean ± Std (window=50)')
        axes[1, 1].grid(True, alpha=0.3)

        path = save_path or os.path.join(self.output_dir, "episode_statistics.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_memory_metrics(
        self,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE or not metrics:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        metrics_to_plot = [
            ('total_chunks', 'Total Chunks', 'blue'),
            ('avg_confidence', 'Avg Confidence', 'green'),
            ('avg_quality', 'Avg Quality', 'orange'),
            ('expired_chunks', 'Expired Chunks', 'red'),
            ('verified_chunks', 'Verified Chunks', 'purple'),
            ('storage_bytes', 'Storage (KB)', 'brown'),
        ]

        for idx, (key, label, color) in enumerate(metrics_to_plot):
            if key in metrics:
                value = metrics[key]
                if key == 'storage_bytes':
                    value = value / 1024
                axes[idx].bar([label], [value], color=color, alpha=0.8)
                axes[idx].set_ylabel(label)
                axes[idx].set_title(label)
                axes[idx].grid(True, alpha=0.3, axis='y')
                for container in axes[idx].containers:
                    axes[idx].text(container.get_x() + container.get_width()/2,
                                 container.get_height(),
                                 f'{value:.1f}', ha='center', va='bottom', fontsize=9)

        for idx in range(len(metrics_to_plot), len(axes)):
            axes[idx].axis('off')

        path = save_path or os.path.join(self.output_dir, "memory_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def create_dashboard(
        self,
        rewards: List[float],
        steps: List[int],
        metrics: Optional[Dict[str, Any]] = None,
        output_name: str = "dashboard.png",
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        smoothed = self._smooth(rewards, 10)
        ax1.plot(rewards, alpha=0.3, label='Raw')
        ax1.plot(smoothed, label='Smoothed', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Learning Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(rewards, bins=25, color='purple', alpha=0.7)
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(steps, color='orange')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.set_title('Episode Length')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(rewards, steps, alpha=0.3, color='green')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Steps')
        ax4.set_title('Reward vs Steps')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 2])
        if metrics:
            keys = ['total_chunks', 'avg_confidence', 'avg_quality']
            values = [metrics.get(k, 0) for k in keys]
            ax5.bar(range(len(keys)), values, color=['blue', 'green', 'orange'])
            ax5.set_xticks(range(len(keys)))
            ax5.set_xticklabels(['Chunks', 'Confidence', 'Quality'], rotation=45)
            ax5.set_ylabel('Value')
            ax5.set_title('Memory Metrics')
            ax5.grid(True, alpha=0.3, axis='y')

        ax6 = fig.add_subplot(gs[2, :])
        window = min(50, len(rewards))
        rolling_mean = self._rolling_mean(rewards, window)
        rolling_std = self._rolling_std(rewards, window)
        x = range(window, len(rewards) + 1)
        ax6.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.3)
        ax6.plot(x, rolling_mean, linewidth=2)
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Reward')
        ax6.set_title(f'Rolling Performance (window={window})')
        ax6.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, output_name)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def _smooth(self, data: List[float], window: int) -> List[float]:
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i + 1]))
        return smoothed

    def _rolling_mean(self, data: List[float], window: int) -> List[float]:
        if len(data) < window:
            return []
        return [np.mean(data[i:i + window]) for i in range(len(data) - window + 1)]

    def _rolling_std(self, data: List[float], window: int) -> List[float]:
        if len(data) < window:
            return []
        return [np.std(data[i:i + window]) for i in range(len(data) - window + 1)]


class MemoryAnalytics:
    """Analytics for TurboMemory usage in RL."""

    def __init__(self, memory_module):
        self.memory = memory_module

    def get_topic_summary(self) -> Dict[str, int]:
        topics = ["experience", "transition", "subgoal", "episode_summary", "failure", "skill"]
        summary = {}
        for topic in topics:
            results = self.memory.query(topic, k=100)
            summary[topic] = len(results)
        return summary

    def get_quality_stats(self) -> Dict[str, float]:
        results = self.memory.query("experience", k=100)
        if not results:
            return {"avg_quality": 0.0, "min_quality": 0.0, "max_quality": 0.0}

        qualities = [r[2].get("quality_score", 0.5) for r in results]
        return {
            "avg_quality": np.mean(qualities),
            "min_quality": np.min(qualities),
            "max_quality": np.max(qualities),
            "std_quality": np.std(qualities),
        }

    def get_memory_efficiency(self) -> Dict[str, Any]:
        results = self.memory.query("experience", k=1000)
        high_quality = sum(1 for r in results if r[2].get("quality_score", 0) > 0.6)
        verified = sum(1 for r in results if r[2].get("verified", False))
        return {
            "total_memories": len(results),
            "high_quality_ratio": high_quality / max(1, len(results)),
            "verified_ratio": verified / max(1, len(results)),
        }

    def export_report(self, output_path: str = "memory_report.json") -> str:
        report = {
            "topic_summary": self.get_topic_summary(),
            "quality_stats": self.get_quality_stats(),
            "efficiency": self.get_memory_efficiency(),
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        return output_path


def create_training_animation(
    rewards: List[float],
    output_path: str = "training_animation.gif",
    interval: int = 50,
) -> Optional[str]:
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(10, 6))

        def update(frame):
            ax.clear()
            end = min((frame + 1) * interval, len(rewards))
            data = rewards[:end]
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - 10 + 1)
                smoothed.append(np.mean(data[start:i + 1]))

            ax.plot(data, alpha=0.3, label='Raw')
            ax.plot(smoothed, label='Smoothed', linewidth=2)
            ax.set_xlim(0, len(rewards))
            ax.set_ylim(min(rewards) - 0.1, max(rewards) + 0.1)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f'Training Progress (Episode {end})')
            ax.legend()
            ax.grid(True, alpha=0.3)

        frames = list(range(0, len(rewards) // interval + 1))
        anim = FuncAnimation(fig, update, frames=frames, interval=100)
        anim.save(output_path, writer=PillowWriter(fps=10))
        plt.close()
        return output_path
    except Exception as e:
        logger.warning(f"Could not create animation: {e}")
        return None
