"""TurboMemory + LangChain RAG pipeline example.

This example shows how to use TurboMemory as a retriever in a LangChain RAG pipeline.

Install: pip install langchain langchain-openai turbomemory
"""

from turbomemory.integrations.langchain import TurboMemoryRetriever


def build_rag_pipeline(root: str = "rag_memory"):
    """Build a RAG pipeline with TurboMemory retriever."""
    retriever = TurboMemoryRetriever(
        root=root,
        k=3,
        enable_verification=True,
    )

    # Add some documents
    from langchain_core.documents import Document

    docs = [
        Document(page_content="TurboMemory uses 4/6/8-bit quantization to compress embeddings", metadata={"topic": "compression"}),
        Document(page_content="Semantic search retrieves memories by embedding similarity", metadata={"topic": "search"}),
        Document(page_content="Consolidation merges duplicates and resolves contradictions", metadata={"topic": "maintenance"}),
        Document(page_content="Quality scoring combines confidence, freshness, and verification", metadata={"topic": "quality"}),
    ]

    retriever.add_documents(docs, topic="rag_docs")

    return retriever


def run_rag_query(retriever, query: str):
    """Run a RAG query."""
    results = retriever.invoke(query)

    print(f"Query: {query}")
    print("-" * 50)

    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Score: {doc.metadata.get('score', 0):.3f}")
        print(f"    Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"    Verified: {doc.metadata.get('verified', False)}")
        print(f"    Content: {doc.page_content}")


if __name__ == "__main__":
    retriever = build_rag_pipeline()

    run_rag_query(retriever, "How does quantization work?")
    print()
    run_rag_query(retriever, "What is consolidation?")

    retriever.close()
