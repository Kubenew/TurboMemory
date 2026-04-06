#!/usr/bin/env python3
"""Tests for RL integration module."""

import pytest
import tempfile
import shutil
from pathlib import Path

from rl_integration import (
    RLMemoryModule,
    HierarchicalRLMemory,
    PrioritizedReplayMemory,
)


class TestRLMemoryModule:
    @pytest.fixture
    def memory(self):
        tmpdir = tempfile.mkdtemp()
        mem = RLMemoryModule(root=tmpdir)
        yield mem
        shutil.rmtree(tmpdir)

    def test_store_experience(self, memory):
        result = memory.store_experience(
            state={"x": 0},
            action=1,
            reward=0.5,
            next_state={"x": 1},
            done=False
        )
        assert result is not None

    def test_store_transition(self, memory):
        result = memory.store_transition(
            state=[0, 0],
            action=0,
            reward=-0.1,
            next_state=[0, 1],
            done=False
        )
        assert result is not None

    def test_store_subgoal(self, memory):
        result = memory.store_subgoal(
            subgoal_label="reach_goal",
            achieved=True,
            state=[5, 5],
            steps_taken=10
        )
        assert result is not None

    def test_store_episode_summary(self, memory):
        result = memory.store_episode_summary(
            episode=0,
            total_reward=10.5,
            steps=100,
            success=True
        )
        assert result is not None
        assert memory.episode_count == 0

    def test_query(self, memory):
        memory.store_experience(state="test", action=1, reward=1.0, next_state="test2", done=False)
        results = memory.query("experience test", k=5)
        assert isinstance(results, list)


class TestHierarchicalRLMemory:
    @pytest.fixture
    def memory(self):
        tmpdir = tempfile.mkdtemp()
        mem = HierarchicalRLMemory(root=tmpdir)
        yield mem
        shutil.rmtree(tmpdir)

    def test_subgoal_lifecycle(self, memory):
        memory.start_subgoal("move_right", target_state=[5, 0])
        assert memory.current_subgoal is not None
        assert memory.current_subgoal["label"] == "move_right"

        result = memory.complete_subgoal(success=True, final_state=[5, 0], steps=5)
        assert result is not None
        assert memory.current_subgoal is None
        assert len(memory.subgoal_history) == 1


class TestPrioritizedReplayMemory:
    @pytest.fixture
    def replay(self):
        tmpdir = tempfile.mkdtemp()
        replay_mem = PrioritizedReplayMemory(root=tmpdir, capacity=100)
        yield replay_mem
        shutil.rmtree(tmpdir)

    def test_push_and_sample(self, replay):
        for i in range(10):
            replay.push(
                state=f"s{i}",
                action=i % 4,
                reward=float(i % 2),
                next_state=f"s{i+1}",
                done=False
            )

        batch = replay.sample(batch_size=5)
        assert len(batch) == 5
        assert all("experience" in item for item in batch)
        assert all("weight" in item for item in batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
