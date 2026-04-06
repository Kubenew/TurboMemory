"""TurboMemory integrations with other frameworks."""

from .retriever import TurboMemoryRetriever
from .chat_history import TurboMemoryChatMessageHistory

# Lazy imports to avoid hard dependencies
def __getattr__(name):
    if name == "LlamaIndexVectorStore":
        from .llamaindex import TurboMemoryVectorStore as LlamaIndexVectorStore
        return LlamaIndexVectorStore
    elif name == "TurboMemoryVectorStore":
        from .langchain import TurboMemoryVectorStore
        return TurboMemoryVectorStore
    elif name == "TurboMemoryEmbeddings":
        from .langchain import TurboMemoryEmbeddings
        return TurboMemoryEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TurboMemoryRetriever",
    "TurboMemoryChatMessageHistory",
    "LlamaIndexVectorStore",
    "TurboMemoryVectorStore", 
    "TurboMemoryEmbeddings",
]