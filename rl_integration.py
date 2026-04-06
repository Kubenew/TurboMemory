#!/usr/bin/env python3
"""RL Memory Module - TurboMemory integration for Reinforcement Learning systems."""

from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
import logging

from turbomemory import TurboMemory, TurboMemoryConfig

logger = logging.getLogger(__name__)


class RLMemoryModule:
    """TurboMemory wrapper for RL systems - stores experiences with semantic search."""

    def __init__(
        self,
        root: str = "rl_memory",
        model_name: str = "all-MiniLM-L6-v2",
        default_confidence: float = 0.8,
    ):
        config = TurboMemoryConfig(
            root=root,
            model_name=model_name,
            enable_exclusions=False,
        )
        self.tm = TurboMemory(config=config)
        self.default_confidence = default_confidence
        self.episode_count = 0

    def store_experience(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        text = self._format_experience(state, action, reward, next_state, done, info)
        return self.tm.add_memory(
            topic="experience",
            text=text,
            confidence=self.default_confidence,
        )

    def store_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool = False,
    ) -> Optional[str]:
        text = f"Transition: state={state}, action={action}, reward={reward:.3f}, next_state={next_state}, done={done}"
        return self.tm.add_memory(topic="transition", text=text)

    def store_subgoal(
        self,
        subgoal_label: str,
        achieved: bool,
        state: Any,
        steps_taken: int = 0,
    ) -> Optional[str]:
        status = "achieved" if achieved else "failed"
        text = f"Subgoal '{subgoal_label}' {status} | state={state} | steps={steps_taken}"
        confidence = 0.9 if achieved else 0.5
        return self.tm.add_memory(
            topic="subgoal",
            text=text,
            confidence=confidence,
        )

    def store_episode_summary(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        success: bool,
        insights: Optional[str] = None,
    ) -> Optional[str]:
        status = "SUCCESS" if success else "FAILED"
        text = f"Episode {episode}: {status} | reward={total_reward:.2f} | steps={steps}"
        if insights:
            text += f" | insights: {insights}"
        self.episode_count = episode
        return self.tm.add_memory(topic="episode_summary", text=text, confidence=0.9)

    def store_failure_case(
        self,
        state: Any,
        action: Any,
        failure_reason: str,
    ) -> Optional[str]:
        text = f"FAILURE at state={state}, action={action}: {failure_reason}"
        return self.tm.add_memory(topic="failure", text=text, confidence=0.95)

    def store_skill_abstraction(
        self,
        skill_name: str,
        preconditions: str,
        effects: str,
        success_rate: float,
    ) -> Optional[str]:
        text = f"Skill '{skill_name}': preconditions={preconditions}, effects={effects}, success_rate={success_rate:.2f}"
        return self.tm.add_memory(topic="skill", text=text, confidence=success_rate)

    def query(self, query_str: str, k: int = 5) -> List[Tuple[float, str, Dict[str, Any]]]:
        return self.tm.query(query_str, k=k)

    def query_with_verification(
        self, query_str: str, k: int = 5
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        return self.tm.verify_and_score(query_str, k=k)

    def query_experiences(self, context: str, k: int = 5) -> List[str]:
        results = self.query(f"experience {context}", k=k)
        return [r[2].get("text", "") for r in results]

    def query_failures(self, situation: str, k: int = 3) -> List[str]:
        results = self.query(f"failure {situation}", k=k)
        return [r[2].get("text", "") for r in results]

    def query_similar_states(self, state: Any, k: int = 3) -> List[Dict[str, Any]]:
        results = self.query(f"state {state}", k=k)
        return [{"text": r[2].get("text", ""), "score": r[0]} for r in results]

    def get_high_reward_transitions(self, k: int = 10) -> List[str]:
        results = self.query("high reward successful transition", k=k)
        return [r[2].get("text", "") for r in results]

    def get_failed_subgoals(self, k: int = 5) -> List[str]:
        results = self.query("subgoal failed", k=k)
        return [r[2].get("text", "") for r in results]

    def _format_experience(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: Optional[Dict[str, Any]],
    ) -> str:
        info_str = f", info={info}" if info else ""
        return f"State: {state} | Action: {action} | Reward: {reward:.4f} | Next: {next_state} | Done: {done}{info_str}"

    def get_metrics(self) -> Dict[str, Any]:
        return self.tm.get_metrics()


class HierarchicalRLMemory(RLMemoryModule):
    """Extended memory module for hierarchical RL with subgoal tracking."""

    def __init__(self, root: str = "hrl_memory", **kwargs):
        super().__init__(root=root, **kwargs)
        self.current_subgoal = None
        self.subgoal_history: List[Dict[str, Any]] = []

    def start_subgoal(self, subgoal_label: str, target_state: Any) -> None:
        self.current_subgoal = {
            "label": subgoal_label,
            "target": target_state,
            "start_time": self.episode_count,
            "attempts": 1,
        }

    def complete_subgoal(self, success: bool, final_state: Any, steps: int) -> Optional[str]:
        if not self.current_subgoal:
            return None
        self.store_subgoal(
            subgoal_label=self.current_subgoal["label"],
            achieved=success,
            state=final_state,
            steps_taken=steps,
        )
        self.subgoal_history.append({
            **self.current_subgoal,
            "success": success,
            "final_state": final_state,
            "steps": steps,
        })
        self.current_subgoal = None
        return self.subgoal_history[-1]["chunk_id"] if success else None

    def query_subgoal_strategies(self, task: str, k: int = 5) -> List[str]:
        results = self.query(f"subgoal achieved {task}", k=k)
        return [r[2].get("text", "") for r in results]

    def get_skill_success_rates(self) -> Dict[str, float]:
        skill_results = self.tm.query("skill abstraction", k=50)
        rates = {}
        for _, _, chunk in skill_results:
            text = chunk.get("text", "")
            if "success_rate=" in text:
                try:
                    parts = text.split("Skill '")[1].split("':")
                    name = parts[0]
                    rate = float(parts[1].split("success_rate=")[1].split(",")[0])
                    rates[name] = rate
                except (IndexError, ValueError):
                    pass
        return rates


class PrioritizedReplayMemory:
    """Memory module with semantic-based prioritized sampling."""

    def __init__(
        self,
        root: str = "prioritized_replay",
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory = RLMemoryModule(root=root)
        self.priorities: List[float] = []
        self.position = 0

    def push(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        self.memory.store_experience(state, action, reward, next_state, done)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
        if len(self.priorities) > self.capacity:
            self.priorities.pop(0)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(probs), size=min(batch_size, len(probs)), p=probs, replace=False)
        weights = (len(probs) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        experiences = self.memory.query_experiences("", k=batch_size)
        return [{"experience": exp, "weight": w, "index": i} for exp, w, i in zip(experiences, weights, indices)]

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority


def integrate_with_training_loop(
    agent,
    env,
    memory_module: RLMemoryModule,
    num_episodes: int,
    query_interval: int = 10,
    query_template: str = "high reward optimal action",
) -> Dict[str, List[float]]:
    """Example integration with RL training loop."""
    rewards_history = []
    insights_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            memory_module.store_experience(state, action, reward, next_state, done, info)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward
            steps += 1

        memory_module.store_episode_summary(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            success=done,
        )
        rewards_history.append(total_reward)

        if episode > 0 and episode % query_interval == 0:
            insights = memory_module.query_experiences(query_template, k=3)
            insights_history.append(insights)
            logger.info(f"Episode {episode}: Retrieved {len(insights)} insights from memory")

    return {"rewards": rewards_history, "insights": insights_history}
