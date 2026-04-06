#!/usr/bin/env python3
"""Tests for visualization module."""

import pytest
import tempfile
import shutil
import os

from visualization import RLVisualizer, MemoryAnalytics


class TestRLVisualizer:
    @pytest.fixture
    def visualizer(self):
        tmpdir = tempfile.mkdtemp()
        viz = RLVisualizer(output_dir=tmpdir)
        yield viz
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_data(self):
        rewards = [i * 0.1 + (hash(i) % 10) * 0.01 for i in range(100)]
        steps = [50 + hash(i) % 20 for i in range(100)]
        return rewards, steps

    def test_plot_learning_curve(self, visualizer, sample_data):
        rewards, _ = sample_data
        path = visualizer.plot_learning_curve(rewards)
        assert path is not None
        assert os.path.exists(path)

    def test_plot_reward_distribution(self, visualizer, sample_data):
        rewards, _ = sample_data
        path = visualizer.plot_reward_distribution(rewards)
        assert path is not None
        assert os.path.exists(path)

    def test_plot_episode_statistics(self, visualizer, sample_data):
        rewards, steps = sample_data
        path = visualizer.plot_episode_statistics(rewards, steps)
        assert path is not None
        assert os.path.exists(path)

    def test_plot_memory_metrics(self, visualizer):
        metrics = {
            "total_chunks": 100,
            "avg_confidence": 0.75,
            "avg_quality": 0.65,
            "expired_chunks": 5,
            "verified_chunks": 20,
            "storage_bytes": 102400,
        }
        path = visualizer.plot_memory_metrics(metrics)
        assert path is not None
        assert os.path.exists(path)

    def test_create_dashboard(self, visualizer, sample_data):
        rewards, steps = sample_data
        metrics = {
            "total_chunks": 50,
            "avg_confidence": 0.8,
            "avg_quality": 0.7,
        }
        path = visualizer.create_dashboard(rewards, steps, metrics)
        assert path is not None
        assert os.path.exists(path)

    def test_smooth_function(self, visualizer):
        data = list(range(100))
        smoothed = visualizer._smooth(data, 10)
        assert len(smoothed) == len(data)
        assert smoothed[0] == 0
        assert smoothed[-1] > smoothed[0]

    def test_rolling_functions(self, visualizer):
        data = list(range(100))
        mean = visualizer._rolling_mean(data, 20)
        std = visualizer._rolling_std(data, 20)
        assert len(mean) == 81
        assert len(std) == 81
        assert all(s >= 0 for s in std)


class TestMemoryAnalytics:
    @pytest.fixture
    def analytics(self):
        from rl_integration import RLMemoryModule
        tmpdir = tempfile.mkdtemp()
        memory = RLMemoryModule(root=tmpdir)
        for i in range(10):
            memory.store_experience(state=f"s{i}", action=0, reward=float(i), next_state=f"s{i+1}", done=False)
        yield MemoryAnalytics(memory)
        shutil.rmtree(tmpdir)

    def test_get_topic_summary(self, analytics):
        summary = analytics.get_topic_summary()
        assert isinstance(summary, dict)
        assert "experience" in summary

    def test_get_quality_stats(self, analytics):
        stats = analytics.get_quality_stats()
        assert "avg_quality" in stats
        assert "min_quality" in stats
        assert "max_quality" in stats

    def test_get_memory_efficiency(self, analytics):
        efficiency = analytics.get_memory_efficiency()
        assert "total_memories" in efficiency
        assert "high_quality_ratio" in efficiency
        assert "verified_ratio" in efficiency

    def test_export_report(self, analytics):
        tmpdir = tempfile.mkdtemp()
        try:
            report_path = analytics.export_report(os.path.join(tmpdir, "report.json"))
            assert os.path.exists(report_path)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
