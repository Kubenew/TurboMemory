"""TurboMemory LangChain retriever integration."""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from turbomemory import TurboMemory


class TurboMemoryRetriever(BaseRetriever):
    """LangChain retriever backed by TurboMemory.

    This retriever uses TurboMemory's quantized semantic search
    with optional verification and quality scoring.

    Example:
        from turbomemory.integrations.langchain import TurboMemoryRetriever

        retriever = TurboMemoryRetriever(
            root="my_memory",
            k=5,
            require_verified=True,
        )

        docs = retriever.invoke("What is TurboQuant?")
        for doc in docs:
            print(doc.page_content)
            print(f"Score: {doc.metadata.get('score')}")
            print(f"Verified: {doc.metadata.get('verified')}")
    """

    tm: TurboMemory
    k: int = 5
    top_topics: int = 5
    min_confidence: float = 0.0
    require_verified: bool = False
    enable_verification: bool = False
    topic_filter: Optional[List[str]] = None

    def __init__(
        self,
        root: str = "turbomemory_data",
        k: int = 5,
        top_topics: int = 5,
        min_confidence: float = 0.0,
        require_verified: bool = False,
        enable_verification: bool = False,
        topic_filter: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        tm = TurboMemory(root=root)
        super().__init__(
            tm=tm,
            k=k,
            top_topics=top_topics,
            min_confidence=min_confidence,
            require_verified=require_verified,
            enable_verification=enable_verification,
            topic_filter=topic_filter,
            **kwargs,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve documents from TurboMemory."""
        if self.enable_verification:
            results = self.tm.verify_and_score(
                query,
                k=self.k,
                top_topics=self.top_topics,
                min_confidence=self.min_confidence,
            )
            documents = []
            for score, topic, chunk, verif in results:
                if self.topic_filter and topic not in self.topic_filter:
                    continue
                if self.require_verified and not verif.verified:
                    continue

                doc = Document(
                    page_content=chunk.get("text", ""),
                    metadata={
                        "score": score,
                        "topic": topic,
                        "chunk_id": chunk.get("chunk_id"),
                        "confidence": chunk.get("confidence"),
                        "staleness": chunk.get("staleness"),
                        "quality_score": chunk.get("quality_score"),
                        "verified": verif.verified,
                        "verification_score": verif.verification_score,
                        "cross_refs": verif.cross_references,
                        "timestamp": chunk.get("timestamp"),
                    },
                )
                documents.append(doc)
        else:
            results = self.tm.query(
                query,
                k=self.k,
                top_topics=self.top_topics,
                min_confidence=self.min_confidence,
                require_verification=self.require_verified,
            )
            documents = []
            for score, topic, chunk in results:
                if self.topic_filter and topic not in self.topic_filter:
                    continue

                doc = Document(
                    page_content=chunk.get("text", ""),
                    metadata={
                        "score": score,
                        "topic": topic,
                        "chunk_id": chunk.get("chunk_id"),
                        "confidence": chunk.get("confidence"),
                        "staleness": chunk.get("staleness"),
                        "quality_score": chunk.get("quality_score"),
                        "verified": chunk.get("verified", False),
                        "timestamp": chunk.get("timestamp"),
                    },
                )
                documents.append(doc)

        return documents

    def add_documents(self, documents: List[Document], topic: str = "default") -> List[str]:
        """Add LangChain documents to TurboMemory."""
        chunk_ids = []
        for doc in documents:
            chunk_id = self.tm.add_memory(
                topic=topic,
                text=doc.page_content,
                confidence=doc.metadata.get("confidence", 0.8),
            )
            if chunk_id:
                chunk_ids.append(chunk_id)
        return chunk_ids

    def close(self) -> None:
        """Close TurboMemory connections."""
        self.tm.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
