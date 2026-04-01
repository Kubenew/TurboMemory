"""TurboMemory + CrewAI integration example.

This example shows how to use TurboMemory as a memory provider for CrewAI agents.

Install: pip install crewai turbomemory
"""

from turbomemory import TurboMemory


class TurboMemoryProvider:
    """Memory provider for CrewAI agents using TurboMemory."""

    def __init__(self, root: str = "crew_memory"):
        self.tm = TurboMemory(root=root)

    def save(self, agent_name: str, content: str, confidence: float = 0.8):
        """Save agent memory."""
        topic = f"agent.{agent_name}"
        return self.tm.add_memory(topic, content, confidence=confidence)

    def query(self, agent_name: str, query: str, k: int = 5):
        """Query agent memory."""
        return self.tm.query(query, k=k, top_topics=3)

    def get_context(self, agent_name: str, query: str, k: int = 3) -> str:
        """Get formatted context string for agent."""
        results = self.query(agent_name, query, k=k)
        if not results:
            return "No relevant memories found."

        context_parts = []
        for score, topic, chunk in results:
            context_parts.append(f"[{topic}] (score: {score:.3f}): {chunk['text']}")

        return "\n".join(context_parts)

    def close(self):
        self.tm.close()


# Example usage
if __name__ == "__main__":
    memory = TurboMemoryProvider(root="crew_demo_memory")

    # Simulate agent memories
    memory.save("researcher", "Python's GIL prevents true parallel threading in CPython")
    memory.save("researcher", "Asyncio provides cooperative multitasking for I/O bound tasks")
    memory.save("writer", "Technical documentation should be clear and concise")
    memory.save("writer", "Use examples to illustrate complex concepts")

    # Query memories
    print("=== Researcher Memories ===")
    context = memory.get_context("researcher", "Python threading")
    print(context)

    print("\n=== Writer Memories ===")
    context = memory.get_context("writer", "documentation best practices")
    print(context)

    memory.close()
