"""LangChain integration for TurboMemory."""

from typing import List, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain.retrievers import BaseRetriever

from turbomemory import TurboMemory
import numpy as np


class TurboMemoryEmbeddings(Embeddings):
    """LangChain Embeddings wrapper using TurboMemory's model."""

    def __init__(
        self,
        tm: Optional[TurboMemory] = None,
        root: str = "turbomemory_data",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self._tm = tm or TurboMemory(root=root, model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        return self._tm.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query."""
        return self._tm.model.encode([text])[0].tolist()


class TurboMemoryVectorStore(VectorStore):
    """LangChain VectorStore backed by TurboMemory."""

    def __init__(
        self,
        tm: Optional[TurboMemory] = None,
        root: str = "turbomemory_data",
        topic: str = "default",
        **kwargs,
    ):
        self._tm = tm or TurboMemory(root=root, **kwargs)
        self._topic = topic

    @property
    def embeddings(self) -> Optional[TurboMemoryEmbeddings]:
        return None

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        **kwargs,
    ) -> List[str]:
        """Add texts to the vector store."""
        chunk_ids = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            chunk_id = self._tm.add_memory(
                topic=self._topic,
                text=text,
                confidence=metadata.get("confidence", 0.8),
                source_ref=metadata.get("source_ref"),
            )
            if chunk_id:
                chunk_ids.append(chunk_id)
        return chunk_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs,
    ) -> List[Document]:
        """Perform similarity search."""
        results = self._tm.query(
            query_text=query,
            k=k,
            min_confidence=filter.get("min_confidence", 0.0) if filter else 0.0,
        )
        
        documents = []
        for score, topic, chunk in results:
            documents.append(Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "topic": topic,
                    "chunk_id": chunk.get("chunk_id"),
                    "confidence": chunk.get("confidence", 0),
                    "score": score,
                },
            ))
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs,
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with scores."""
        results = self._tm.query(query_text=query, k=k)
        
        docs_with_scores = []
        for score, topic, chunk in results:
            doc = Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "topic": topic,
                    "chunk_id": chunk.get("chunk_id"),
                    "confidence": chunk.get("confidence", 0),
                },
            )
            docs_with_scores.append((doc, score))
        return docs_with_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict]] = None,
        **kwargs,
    ) -> "TurboMemoryVectorStore":
        """Create vector store from texts."""
        store = cls(**kwargs)
        store.add_texts(texts, metadatas)
        return store


class TurboMemoryRetriever(BaseRetriever):
    """LangChain Retriever backed by TurboMemory."""

    def __init__(
        self,
        tm: Optional[TurboMemory] = None,
        root: str = "turbomemory_data",
        topic: str = "default",
        k: int = 4,
        **kwargs,
    ):
        self._tm = tm or TurboMemory(root=root, **kwargs)
        self._topic = topic
        self._k = k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents."""
        results = self._tm.query(query_text=query, k=self._k)
        
        documents = []
        for score, topic, chunk in results:
            documents.append(Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "topic": topic,
                    "chunk_id": chunk.get("chunk_id"),
                    "score": score,
                },
            ))
        return documents


def get_turbo_memory_vectorstore(
    root: str = "turbomemory_data",
    topic: str = "default",
    **kwargs,
) -> TurboMemoryVectorStore:
    """Get a TurboMemory vector store."""
    return TurboMemoryVectorStore(root=root, topic=topic, **kwargs)