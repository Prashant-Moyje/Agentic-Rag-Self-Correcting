"""
Load a PDF, split into chunks, and write the embeddings into both
Pinecone and Milvus so multi_store_search() has something to find.

Usage:
    python -m src.ingestion --pdf path/to/report.pdf
"""
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vector_stores import upsert_documents


def ingest(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
    print(f"Loading {pdf_path}...")
    documents = PyPDFLoader(pdf_path).load()
    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    upsert_documents(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF to ingest")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    ingest(args.pdf, args.chunk_size, args.chunk_overlap)