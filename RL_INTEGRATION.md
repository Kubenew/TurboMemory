# TurboMemory RL Integration

Integration scaffolding for combining TurboMemory (semantic memory) with Hierarchical Reinforcement Learning systems.

## Architecture

```
+-----------------+      +--------+---------+      +-----------------+
| HRL Environment | <--> |    RL Controller  | <--> |  Memory Module  |
| (Grid, tasks)   |      | (hrl-minimal / ASI)|     | (TurboMemory)   |
+-----------------+      +-------------------+      +-----------------+
```

## Quick Start

```bash
cd TurboMemory
pip install -e .
```

## Usage

### Basic RL Memory Module

```python
from rl_integration import RLMemoryModule

memory = RLMemoryModule(root="my_rl_memory")

# Store experiences
memory.store_experience(
    state={"position": [0, 0]},
    action=1,
    reward=0.5,
    next_state={"position": [0, 1]},
    done=False
)

# Query similar experiences
insights = memory.query_experiences("high reward optimal action", k=5)
```

### Hierarchical RL with Subgoals

```python
from rl_integration import HierarchicalRLMemory

memory = HierarchicalRLMemory(root="hrl_memory")

# Track subgoals
memory.start_subgoal("reach_door", target_state=[5, 5])
# ... perform actions ...
memory.complete_subgoal(success=True, final_state=[5, 5], steps=10)

# Query strategies
strategies = memory.query_subgoal_strategies("navigation planning", k=3)
```

### Prioritized Replay

```python
from rl_integration import PrioritizedReplayMemory

replay = PrioritizedReplayMemory(capacity=10000)

# Store transitions
replay.push(state, action, reward, next_state, done)

# Sample with semantic prioritization
batch = replay.sample(batch_size=32)
```

## Running Examples

```bash
# hrl-minimal style
python examples/hrl_minimal_example.py

# ASI style
python examples/asi_example.py

# Visualization demo
pip install matplotlib
python examples/visualization_example.py
```

## Visualization

The `visualization.py` module provides comprehensive plotting and analytics:

```python
from visualization import RLVisualizer, MemoryAnalytics

visualizer = RLVisualizer(output_dir="plots")

# Learning curves
visualizer.plot_learning_curve(rewards, window_size=10)

# Dashboard with all metrics
visualizer.create_dashboard(rewards, steps, metrics)

# Subgoal success rates
visualizer.plot_subgoal_success_rate(subgoal_history)

# Memory analytics
analytics = MemoryAnalytics(memory_module)
report = analytics.export_report("report.json")
```

**Available plots:**
| Method | Description |
|--------|-------------|
| `plot_learning_curve()` | Episode rewards with smoothing |
| `plot_reward_distribution()` | Histogram, box plot, percentiles |
| `plot_episode_statistics()` | Combined reward/step analysis |
| `plot_subgoal_success_rate()` | Success rates per subgoal |
| `plot_memory_metrics()` | Memory quality/confidence stats |
| `create_dashboard()` | Complete training dashboard |
| `create_training_animation()` | GIF of training progress |

## Modules

- `rl_integration.py` - Core integration classes
- `visualization.py` - Plotting and analytics
- `examples/` - Integration examples

## Integration Complexity

| Feature | Complexity |
|---------|------------|
| Store RL experiences | Low |
| Use semantic queries for guidance | Medium |
| Memory-driven hierarchical decisions | High |
