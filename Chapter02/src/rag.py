"""
Main RAG System Interface
Provides a simple API for ingesting PDFs and querying them.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.ingest import DocumentIngestor
from src.query import RAGQueryEngine


class RAGSystem:
    """Main interface for the RAG system."""

    def __init__(self, persist_directory: str = "./chroma_db", top_k: int = 3):
        """
        Initialize the RAG system.

        Args:
            persist_directory: Directory to persist ChromaDB data
            top_k: Number of relevant chunks to retrieve for queries
        """
        self.persist_directory = persist_directory
        self.top_k = top_k
        self.ingestor = DocumentIngestor(persist_directory=persist_directory)
        self.query_engine = None

    def ingest_file(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50, force: bool = False) -> None:
        """
        Ingest a single file (PDF, TXT, MD) into the vector database.
        Skips if already ingested unless force=True.

        Args:
            file_path: Path to the file
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            force: Force re-ingestion even if already processed
        """
        self.ingestor.ingest_file(file_path, chunk_size, chunk_overlap, force)
        # Reset query engine to reload vector store
        self.query_engine = None

    def ingest_folder(self, folder_path: str, chunk_size: int = 500, chunk_overlap: int = 50, force: bool = False) -> None:
        """
        Ingest all supported files in a folder.
        Only processes new or modified files unless force=True.

        Args:
            folder_path: Path to folder containing documents
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            force: Force re-ingestion of all files
        """
        self.ingestor.ingest_folder(folder_path, chunk_size, chunk_overlap, force=force)
        # Reset query engine to reload vector store
        self.query_engine = None

    def ingest_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Ingest a PDF file (backward compatibility method).

        Args:
            pdf_path: Path to the PDF file
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.ingest_file(pdf_path, chunk_size, chunk_overlap)

    def is_vectorstore_ready(self) -> bool:
        return os.path.exists(self.persist_directory)

    def ask(self, question: str, verbose: bool = False) -> Dict[str, any]:
        """
        Ask a question using the RAG system.

        Args:
            question: User's question
            verbose: If True, print detailed information

        Returns:
            Dictionary containing answer, sources, and question

        Raises:
            RuntimeError: If vector store hasn't been created yet
        """
        if not self.is_vectorstore_ready():
            raise RuntimeError(
                "Vector store not found. Please ingest a PDF first using ingest_pdf()"
            )

        # Lazy load query engine
        if self.query_engine is None:
            self.query_engine = RAGQueryEngine(
                persist_directory=self.persist_directory,
                top_k=self.top_k
            )

        return self.query_engine.query(question, verbose=verbose)

    def get_relevant_chunks(self, question: str, k: Optional[int] = None) -> list:
        """
        Get relevant chunks without generating an answer.

        Args:
            question: User's question
            k: Number of chunks to retrieve

        Returns:
            List of relevant chunks with metadata
        """
        if not self.is_vectorstore_ready():
            raise RuntimeError(
                "Vector store not found. Please ingest documents first using ingest_folder() or ingest_file()"
            )

        if self.query_engine is None:
            self.query_engine = RAGQueryEngine(
                persist_directory=self.persist_directory,
                top_k=self.top_k
            )

        return self.query_engine.get_relevant_chunks(question, k)

    def get_ingested_files(self) -> dict:
        return self.ingestor.get_ingested_files()


def main():
    """Example usage demonstrating the complete RAG workflow."""
    print("\nRAG System Demo\n")

    # Initialize RAG system
    rag = RAGSystem(persist_directory="./chroma_db", top_k=3)

    # Ingest all files in data folder (only new/modified files)
    print("Ingesting documents from ./data folder...\n")
    rag.ingest_folder("./data")

    # Show ingested files
    ingested = rag.get_ingested_files()
    if ingested:
        print(f"\nTotal ingested files: {len(ingested)}")
        for file_path in ingested:
            print(f"   - {Path(file_path).name}")

    print("\nReady for queries!\n")

    # Example questions
    questions = [
        "What is DataMind?",
        "How can I get support?",
        "What are the pricing options?"
    ]

    print("\nRunning example queries:\n")

    for i, question in enumerate(questions, 1):
        print(f"\nQuery {i}: {question}")

        try:
            result = rag.ask(question, verbose=False)

            print(f"\nAnswer:")
            print(result['answer'])

            print(f"\nSources:")
            for j, source in enumerate(result['sources'], 1):
                print(f"  [{j}] Page {source['page']}")

        except Exception as e:
            print(f"Error: {e}")

    print("\nDemo complete.\n")


if __name__ == "__main__":
    main()
