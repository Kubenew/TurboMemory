#!/usr/bin/env python3
"""Example demonstrating visualization and analytics for RL + TurboMemory."""

import numpy as np
from rl_integration import RLMemoryModule, HierarchicalRLMemory
from visualization import RLVisualizer, MemoryAnalytics, create_training_animation


def generate_synthetic_training_data(num_episodes: int = 500):
    rewards = []
    steps = []

    base_reward = -1.0
    improvement_rate = 0.01

    for episode in range(num_episodes):
        if episode < 100:
            reward = base_reward + np.random.randn() * 0.5
        else:
            progress = min(1.0, (episode - 100) / 300)
            reward = base_reward + progress * 3 + np.random.randn() * 0.3

        step_count = int(100 - progress * 50 + np.random.randint(-10, 10))
        step_count = max(20, min(100, step_count))

        rewards.append(reward)
        steps.append(step_count)

    return rewards, steps


def main():
    print("=== RL + TurboMemory Visualization Demo ===\n")

    tmpdir = "visualization_demo"
    memory = HierarchicalRLMemory(root=f"{tmpdir}/memory")
    visualizer = RLVisualizer(output_dir=f"{tmpdir}/plots")

    num_episodes = 300
    print(f"Simulating {num_episodes} training episodes...")

    rewards, steps = generate_synthetic_training_data(num_episodes)

    for episode in range(num_episodes):
        state = {"pos": np.random.randint(0, 10)}
        action = np.random.randint(0, 4)
        reward = rewards[episode]
        next_state = {"pos": np.random.randint(0, 10)}
        done = episode > 50 and np.random.random() < 0.3

        memory.store_experience(state, action, reward, next_state, done)

        if episode % 10 == 0:
            memory.store_episode_summary(
                episode=episode,
                total_reward=reward,
                steps=steps[episode],
                success=done,
            )

        if episode < 50 and np.random.random() < 0.1:
            memory.store_subgoal(
                subgoal_label=np.random.choice(["move_right", "move_up", "reach_goal"]),
                achieved=done,
                state=state,
                steps_taken=steps[episode] // 4,
            )

        if episode % 50 == 0:
            memory.store_failure_case(state, action, "timeout" if not done else "success")

    print("Generating visualizations...\n")

    print("1. Learning Curve")
    path = visualizer.plot_learning_curve(rewards, window_size=20)
    print(f"   Saved: {path}")

    print("2. Reward Distribution")
    path = visualizer.plot_reward_distribution(rewards)
    print(f"   Saved: {path}")

    print("3. Episode Statistics")
    path = visualizer.plot_episode_statistics(rewards, steps)
    print(f"   Saved: {path}")

    print("4. Memory Metrics")
    metrics = memory.tm.get_metrics()
    path = visualizer.plot_memory_metrics(metrics.to_dict())
    print(f"   Saved: {path}")

    print("5. Subgoal Success Rates")
    if memory.subgoal_history:
        path = visualizer.plot_subgoal_success_rate(memory.subgoal_history)
        print(f"   Saved: {path}")
    else:
        print("   No subgoal history to plot")

    print("6. Complete Dashboard")
    path = visualizer.create_dashboard(rewards, steps, metrics.to_dict())
    print(f"   Saved: {path}")

    print("\n--- Memory Analytics ---")
    analytics = MemoryAnalytics(memory)

    print("\nTopic Summary:")
    for topic, count in analytics.get_topic_summary().items():
        print(f"  {topic}: {count}")

    print("\nQuality Stats:")
    for key, value in analytics.get_quality_stats().items():
        print(f"  {key}: {value:.3f}")

    print("\nMemory Efficiency:")
    for key, value in analytics.get_memory_efficiency().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    report_path = analytics.export_report(f"{tmpdir}/memory_report.json")
    print(f"\nExported report: {report_path}")

    print("\n=== Visualization Complete ===")
    print(f"Check the '{tmpdir}/plots/' directory for generated images.")


if __name__ == "__main__":
    main()
