#!/usr/bin/env python3
"""Integration example: TurboMemory with hierarchical RL (hrl-minimal style)."""

import numpy as np
import random
from typing import List, Tuple, Any, Optional, Dict
from dataclasses import dataclass

from rl_integration import HierarchicalRLMemory, RLMemoryModule


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class GridWorldEnv:
    def __init__(self, size: int = 5):
        self.size = size
        self.state = np.array([0, 0])
        self.goal = np.array([size - 1, size - 1])
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.max_steps = 100

    def reset(self) -> np.ndarray:
        self.state = np.array([0, 0])
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        x, y = self.state
        dx, dy = self.actions[action]
        nx, ny = np.clip(x + dx, 0, self.size - 1), np.clip(y + dy, 0, self.size - 1)
        self.state = np.array([nx, ny])

        reward = -0.01
        done = False
        info = {}

        if np.array_equal(self.state, self.goal):
            reward = 1.0
            done = True
            info["success"] = True

        steps = info.get("steps", 0) + 1
        info["steps"] = steps

        return self.state.copy(), reward, done, info

    @property
    def observation_space(self):
        return (self.size, self.size)


class HierarchicalAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        subgoals: List[str],
        memory: HierarchicalRLMemory,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.subgoals = subgoals
        self.memory = memory
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.lr = 0.1
        self.gamma = 0.95

    def state_to_key(self, state: np.ndarray) -> str:
        return tuple(int(x) for x in state)

    def select_subgoal(self, state: np.ndarray, goal: np.ndarray) -> Tuple[str, np.ndarray]:
        state_key = self.state_to_key(state)
        query_results = self.memory.query_subgoal_strategies(
            f"reaching goal at {tuple(goal)} from {state_key}", k=3
        )
        if query_results and random.random() < 0.3:
            return self.subgoals[0], goal
        dist = np.abs(goal - state)
        if dist[0] >= dist[1]:
            return "move_right" if state[0] < goal[0] else "move_left"
        return "move_down" if state[1] < goal[1] else "move_up"

    def act(self, state: np.ndarray, goal: Optional[np.ndarray] = None) -> int:
        if goal is not None:
            subgoal, _ = self.select_subgoal(state, goal)
            self.memory.start_subgoal(subgoal, goal)

        key = self.state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.n_actions)

        td_target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.lr * (td_target - self.q_table[key][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_hrl_with_memory(
    num_episodes: int = 500,
    env_size: int = 5,
    memory_root: str = "hrl_minimal_memory",
):
    env = GridWorldEnv(size=env_size)
    memory = HierarchicalRLMemory(root=memory_root)
    goal = np.array([env_size - 1, env_size - 1])

    subgoals = ["move_right", "move_left", "move_up", "move_down"]
    agent = HierarchicalAgent(
        state_dim=2,
        n_actions=len(env.actions),
        subgoals=subgoals,
        memory=memory,
    )

    rewards_history = []
    success_count = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        subgoal_steps = 0
        current_subgoal = None

        while steps < env.max_steps:
            action = agent.act(state, goal=goal)
            next_state, reward, done, info = env.step(action)

            memory.store_transition(state, action, reward, next_state, done)
            agent.learn(state, action, reward, next_state)

            if current_subgoal:
                subgoal_steps += 1
                if subgoal_steps > 20:
                    memory.store_subgoal(current_subgoal, achieved=False, state=state)
                    current_subgoal = None
                    subgoal_steps = 0

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                success_count += 1
                if current_subgoal:
                    memory.store_subgoal(current_subgoal, achieved=True, state=state, steps_taken=subgoal_steps)
                break

        if not done and current_subgoal:
            memory.store_subgoal(current_subgoal, achieved=False, state=state, steps_taken=subgoal_steps)

        memory.store_episode_summary(
            episode=episode,
            total_reward=episode_reward,
            steps=steps,
            success=done,
        )

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if episode % 50 == 0:
            insights = memory.query_subgoal_strategies("subgoal planning", k=3)
            print(f"Episode {episode}: Success rate={success_count/50:.2%}, Insights={len(insights)}")
            success_count = 0

    return rewards_history


if __name__ == "__main__":
    train_hrl_with_memory(num_episodes=200)
