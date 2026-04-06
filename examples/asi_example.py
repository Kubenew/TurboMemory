#!/usr/bin/env python3
"""Integration example: TurboMemory with ASI-style hierarchical RL."""

import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass
import random

from rl_integration import RLMemoryModule, HierarchicalRLMemory


@dataclass
class Experience:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    meta: Optional[Dict] = None


class SimpleMDP:
    def __init__(self, n_states: int = 10, n_actions: int = 4):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_probs = np.random.rand(n_states, n_actions, n_states)
        self.transition_probs = self.transition_probs / self.transition_probs.sum(axis=2, keepdims=True)
        self.rewards = np.random.randn(n_states, n_actions)

    def reset(self) -> int:
        return 0

    def step(self, state: int, action: int) -> Tuple[int, float, bool, Dict]:
        probs = self.transition_probs[state, action]
        next_state = np.random.choice(self.n_states, p=probs)
        reward = float(self.rewards[state, action])
        done = next_state == self.n_states - 1
        return next_state, reward, done, {"state": state, "action": action}


class Skill:
    def __init__(self, name: str, preconditions: str, effects: str):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects
        self.success_count = 0
        self.failure_count = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def attempt(self, state: Any, env) -> Tuple[bool, Any]:
        return False, state


class MoveSkill(Skill):
    def __init__(self, direction: str, dx: int, dy: int):
        super().__init__(
            name=f"move_{direction}",
            preconditions=f"can move {direction}",
            effects=f"position changes by ({dx}, {dy})",
        )
        self.dx = dx
        self.dy = dy

    def attempt(self, state: Any, env) -> Tuple[bool, Any]:
        if isinstance(state, (list, tuple)):
            x, y = state[0], state[1]
            new_state = (max(0, min(x + self.dx, 4)), max(0, min(y + self.dy, 4)))
            success = new_state != (x, y)
            return success, new_state
        return False, state


class HighLevelPolicy:
    def __init__(self, skills: List[Skill], memory: HierarchicalRLMemory):
        self.skills = skills
        self.memory = memory
        self.q_values = {s.name: 0.0 for s in skills}

    def select_skill(self, state: Any, goal: Any) -> Skill:
        for skill in self.skills:
            memory_results = self.memory.query_similar_states(state, k=2)
            if memory_results and random.random() < 0.2:
                best_skill = random.choice(self.skills)
                self.memory.store_skill_abstraction(
                    skill_name=best_skill.name,
                    preconditions=best_skill.preconditions,
                    effects=best_skill.effects,
                    success_rate=best_skill.success_rate,
                )
                return best_skill

        return random.choice(self.skills)

    def update(self, skill_name: str, reward: float):
        self.q_values[skill_name] += 0.1 * (reward - self.q_values[skill_name])


class LowLevelPolicy:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.q_table = {}

    def select_action(self, state: Any) -> int:
        if isinstance(state, (list, tuple)):
            key = tuple(state)
        else:
            key = state

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)

        return int(np.argmax(self.q_table[key])) if random.random() > 0.1 else random.randint(0, self.n_actions - 1)

    def update(self, state: Any, action: int, reward: float, next_state: Any):
        if isinstance(state, (list, tuple)):
            key = tuple(state)
        else:
            key = state

        if isinstance(next_state, (list, tuple)):
            next_key = tuple(next_state)
        else:
            next_key = next_state

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.n_actions)

        td_error = reward + 0.95 * np.max(self.q_table[next_key]) - self.q_table[key][action]
        self.q_table[key][action] += 0.1 * td_error


def train_asi_with_memory(
    num_episodes: int = 300,
    memory_root: str = "asi_memory",
):
    env = SimpleMDP(n_states=10, n_actions=4)
    memory = HierarchicalRLMemory(root=memory_root)

    skills = [
        MoveSkill("right", 1, 0),
        MoveSkill("left", -1, 0),
        MoveSkill("up", 0, 1),
        MoveSkill("down", 0, -1),
    ]

    high_level = HighLevelPolicy(skills, memory)
    low_level = LowLevelPolicy(n_actions=4)

    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        skill_attempts = 0

        while steps < 100:
            skill = high_level.select_skill(state, goal=env.n_states - 1)
            skill_success, next_state = skill.attempt(state, env)

            if not skill_success:
                action = low_level.select_action(state)
                next_state, reward, done, info = env.step(state, action)
                low_level.update(state, action, reward, next_state)
            else:
                reward = 0.1

            memory.store_transition(state, action if not skill_success else skill.name, reward, next_state, done)

            if skill_success:
                skill.success_count += 1
            else:
                skill.failure_count += 1

            high_level.update(skill.name, reward)

            state = next_state
            episode_reward += reward
            steps += 1
            skill_attempts += 1

            if done:
                break

        memory.store_episode_summary(
            episode=episode,
            total_reward=episode_reward,
            steps=steps,
            success=done,
            insights=f"skill_attempts={skill_attempts}",
        )

        if episode % 10 == 0:
            skill_rates = memory.get_skill_success_rates()
            failed = memory.get_failed_subgoals(k=3)
            print(f"Episode {episode}: reward={episode_reward:.2f}, failed_subgoals={len(failed)}")

        rewards_history.append(episode_reward)

    return rewards_history


if __name__ == "__main__":
    train_asi_with_memory(num_episodes=100)
