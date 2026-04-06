"""LlamaIndex integration for TurboMemory."""

from typing import List, Dict, Any, Optional
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    VectorStore,
    Document,
)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import BaseQueryEngine

from turbomemory import TurboMemory


class TurboMemoryVectorStore(VectorStore):
    """LlamaIndex VectorStore backed by TurboMemory."""

    def __init__(
        self,
        tm: Optional[TurboMemory] = None,
        root: str = "turbomemory_data",
        **kwargs,
    ):
        self._tm = tm or TurboMemory(root=root, **kwargs)
        self._vector_store = "TurboMemory"

    @property
    def client(self) -> TurboMemory:
        return self._tm

    def add(self, nodes: List[BaseNode], **add_kwargs) -> None:
        """Add nodes to the vector store."""
        for node in nodes:
            text = node.get_content()
            doc_id = node.id_
            metadata = node.metadata or {}
            topic = metadata.get("topic", "default")
            
            self._tm.add_memory(
                topic=topic,
                text=text,
                source_ref=doc_id,
                confidence=metadata.get("confidence", 0.8),
            )

    def delete(self, ref_doc_id: str, **delete_kwargs) -> None:
        """Delete nodes by ref_doc_id."""
        # Implementation would need chunk-level delete
        pass

    def query(self, query: str, **kwargs) -> Any:
        """Query the vector store."""
        k = kwargs.get("similarity_top_k", 5)
        results = self._tm.query(query_text=query, k=k)
        
        # Convert to LlamaIndex format
        from llama_index.core.schema import NodeWithScore
        nodes = []
        for score, topic, chunk in results:
            node = TextNode(
                text=chunk.get("text", ""),
                id_=chunk.get("chunk_id", ""),
                metadata={"topic": topic, "confidence": chunk.get("confidence", 0)},
            )
            nodes.append(NodeWithScore(node=node, score=score))
        
        return nodes

    def persist(self, persist_path: str, **kwargs) -> None:
        """Persist the vector store."""
        pass  # TurboMemory handles persistence automatically


class TurboMemoryIndex:
    """TurboMemory-backed LlamaIndex index."""

    def __init__(
        self,
        tm: Optional[TurboMemory] = None,
        root: str = "turbomemory_data",
        **kwargs,
    ):
        self._tm = tm or TurboMemory(root=root, **kwargs)
        self._vector_store = TurboMemoryVectorStore(tm=self._tm)
        self._index: Optional[VectorStoreIndex] = None

    def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create index from documents."""
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        return self._index

    def as_retriever(self, **kwargs) -> BaseRetriever:
        """Get a retriever from the index."""
        if self._index is None:
            raise ValueError("Index not created. Call create_index first.")
        return self._index.as_retriever(**kwargs)

    def as_query_engine(self, **kwargs) -> BaseQueryEngine:
        """Get a query engine from the index."""
        if self._index is None:
            raise ValueError("Index not created. Call create_index first.")
        return self._index.as_query_engine(**kwargs)


def getTurboMemoryIndex(
    root: str = "turbomemory_data",
    **kwargs,
) -> TurboMemoryIndex:
    """Get or create a TurboMemory-backed LlamaIndex index."""
    return TurboMemoryIndex(root=root, **kwargs)