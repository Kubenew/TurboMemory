"""TurboMemory v3 - Public API wrapper."""

from .kernel import TurboMemoryKernel
from .config import TurboMemoryConfig

class TurboMemory:
    """TurboMemory v3 - Persistent Agent Memory OS."""
    
    def __init__(
        self,
        root: str = "./tm_store",
        config: TurboMemoryConfig = None,
    ):
        self.config = config or TurboMemoryConfig()
        self.kernel = TurboMemoryKernel(root=root, config=self.config)
    
    def add(
        self,
        text: str,
        tags = None,
        topic = None,
        confidence: float = 0.5,
        ttl_seconds = None,
        source: str = "manual",
        agent_id: str = "default",
        extra = None,
    ):
        """Add a memory."""
        return self.kernel.add(
            text=text,
            tags=tags,
            topic=topic,
            confidence=confidence,
            ttl_seconds=ttl_seconds,
            source=source,
            agent_id=agent_id,
            extra=extra,
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        tags_any = None,
        source = None,
        min_confidence = None,
        topic = None,
        verify: bool = False,
    ):
        """Search for memories."""
        return self.kernel.search(
            query=query,
            top_k=top_k,
            tags_any=tags_any,
            source=source,
            min_confidence=min_confidence,
            topic=topic,
            verify=verify,
        )
    
    def reinforce(self, mem_id: int, delta: float = 0.1):
        """Increase confidence."""
        return self.kernel.reinforce(mem_id, delta)
    
    def penalize(self, mem_id: int, delta: float = 0.2):
        """Decrease confidence."""
        return self.kernel.penalize(mem_id, delta)
    
    def forget(self, mem_id: int):
        """Soft forget."""
        return self.kernel.forget(mem_id)
    
    def delete(self, mem_id: int):
        """Hard delete."""
        return self.kernel.delete(mem_id)
    
    def stats(self):
        """Get statistics."""
        return self.kernel.stats()
    
    def get(self, mem_id: int):
        """Get a memory by ID."""
        return self.kernel.get_memory(mem_id)
    
    def close(self):
        """Close resources."""
        return self.kernel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()