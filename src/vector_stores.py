"""
Wrappers around Pinecone and Milvus so the rest of the app can treat
'search across multiple vector databases' as one function call.
"""
from __future__ import annotations

from typing import List
from langchain_core.documents import Document

from . import config


def get_embeddings():
    """Return the configured embedding model (OpenAI or local HF)."""
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)


def get_pinecone_store():
    """Connect to (or create) the Pinecone index and return a LangChain VectorStore."""
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing:
        embeddings = get_embeddings()
        dim = len(embeddings.embed_query("dimension probe"))
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )

    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=config.PINECONE_API_KEY,
    )


def get_milvus_store():
    """Connect to (or create) the Milvus collection and return a LangChain VectorStore."""
    from langchain_milvus import Milvus

    return Milvus(
        embedding_function=get_embeddings(),
        collection_name=config.MILVUS_COLLECTION_NAME,
        connection_args={"uri": config.MILVUS_URI, "token": config.MILVUS_TOKEN},
        auto_id=True,
    )


def _sanitize_for_milvus(chunks: List[Document]) -> List[Document]:
    """Milvus reserves certain metadata field names (e.g. 'text') internally.
    Strip any colliding keys so add_documents doesn't error out."""
    reserved = {"text", "pk", "vector"}
    cleaned = []
    for doc in chunks:
        new_metadata = {k: v for k, v in doc.metadata.items() if k not in reserved}
        cleaned.append(Document(page_content=doc.page_content, metadata=new_metadata))
    return cleaned


def upsert_documents(chunks: List[Document]) -> None:
    """Embed and write the same chunks into both Pinecone and Milvus."""
    print(f"Upserting {len(chunks)} chunks into Pinecone...")
    get_pinecone_store().add_documents(chunks)

    print(f"Upserting {len(chunks)} chunks into Milvus...")
    milvus_chunks = _sanitize_for_milvus(chunks)
    get_milvus_store().add_documents(milvus_chunks)

    print("Done. Both vector stores are in sync.")


def multi_store_search(query: str, k: int = config.RETRIEVAL_K) -> List[Document]:
    """
    Search Pinecone and Milvus in parallel, merge, and de-duplicate results.
    """
    import concurrent.futures

    def _search_pinecone():
        try:
            return get_pinecone_store().similarity_search(query, k=k)
        except Exception as e:
            print(f"[warn] Pinecone search failed: {e}")
            return []

    def _search_milvus():
        try:
            return get_milvus_store().similarity_search(query, k=k)
        except Exception as e:
            print(f"[warn] Milvus search failed: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        pinecone_future = executor.submit(_search_pinecone)
        milvus_future = executor.submit(_search_milvus)
        pinecone_docs = pinecone_future.result()
        milvus_docs = milvus_future.result()

    merged = pinecone_docs + milvus_docs
    seen = set()
    deduped: List[Document] = []
    for doc in merged:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            deduped.append(doc)

    return deduped