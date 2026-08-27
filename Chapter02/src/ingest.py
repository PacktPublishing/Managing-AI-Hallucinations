"""
Document Ingestion Module for RAG System
Handles PDF, TXT, and other document parsing, chunking, embedding generation, and vector storage.
Tracks ingested files to avoid re-processing.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional
from dotenv import load_dotenv

# Add project root to path when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class DocumentIngestor:
    """Handles multi-format document ingestion into ChromaDB vector store with file tracking."""

    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the document ingestor.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.persist_directory = persist_directory
        self.tracking_file = os.path.join(persist_directory, "ingested_files.json")

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.api_key
        )

    def _get_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_tracking_data(self) -> Dict[str, str]:
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_tracking_data(self, tracking_data: Dict[str, str]) -> None:
        os.makedirs(self.persist_directory, exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)

    def _is_file_ingested(self, file_path: str) -> bool:
        tracking_data = self._load_tracking_data()
        file_path = str(Path(file_path).resolve())

        if file_path not in tracking_data:
            return False

        current_hash = self._get_file_hash(file_path)
        return tracking_data[file_path] == current_hash

    def _mark_file_ingested(self, file_path: str) -> None:
        tracking_data = self._load_tracking_data()
        resolved_path = str(Path(file_path).resolve())

        # Check if file exists before hashing
        if not os.path.exists(resolved_path):
            print(f"Warning: File not found for tracking: {resolved_path}")
            return

        file_hash = self._get_file_hash(resolved_path)
        tracking_data[resolved_path] = file_hash
        self._save_tracking_data(tracking_data)

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on file extension.

        Args:
            file_path: Path to document file

        Returns:
            List of Document objects with content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext in {'.txt', '.md'}:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: {self.SUPPORTED_EXTENSIONS}")

        documents = loader.load()

        # Add source file to metadata
        for doc in documents:
            doc.metadata['source_file'] = Path(file_path).name
            doc.metadata['file_type'] = file_ext

        print(f"Loaded {len(documents)} page(s)/section(s) from {Path(file_path).name}")
        return documents

    def chunk_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        Split documents into chunks with overlap.

        Args:
            documents: List of Document objects
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks

        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        return chunks

    def _delete_file_chunks(self, file_name: str) -> None:
        """
        Delete all chunks associated with a specific file from the vector store.

        Args:
            file_name: Name of the file whose chunks should be deleted
        """
        if not os.path.exists(self.persist_directory):
            return

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        # Get all documents with this source file
        try:
            results = vectorstore.get(where={"source_file": file_name})
            if results and results['ids']:
                vectorstore.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} old chunks from {file_name}")
        except Exception as e:
            print(f"Warning: Could not delete old chunks: {e}")

    def create_or_update_vector_store(self, chunks: List[Document]) -> Chroma:
        """
        Create or update ChromaDB vector store with document chunks.

        Args:
            chunks: List of chunked Document objects

        Returns:
            Chroma vector store instance
        """
        print(f"Generating embeddings and storing in ChromaDB...")

        if os.path.exists(self.persist_directory):
            # Load existing and add new chunks
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            vectorstore.add_documents(chunks)
            print(f"Added {len(chunks)} chunks to existing vector store")
        else:
            # Create new vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Created new vector store with {len(chunks)} chunks")

        print(f"Persisted to: {self.persist_directory}")
        return vectorstore

    def ingest_file(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50, force: bool = False) -> Optional[Chroma]:
        """
        Ingest a single file: load → chunk → embed → store.
        Skips if already ingested unless force=True.
        If file was modified, deletes old chunks before adding new ones.

        Args:
            file_path: Path to file
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            force: Force re-ingestion even if file already processed

        Returns:
            Chroma vector store instance or None if skipped
        """
        file_path = str(Path(file_path).resolve())
        file_name = Path(file_path).name

        # Check if already ingested
        is_already_ingested = self._is_file_ingested(file_path)
        tracking_data = self._load_tracking_data()
        is_tracked = file_path in tracking_data

        if not force and is_already_ingested:
            print(f"Skipping {file_name} (already ingested)")
            return None

        # If file was previously ingested but hash changed, delete old chunks
        if is_tracked and not is_already_ingested:
            print(f"\nRe-ingesting modified file: {file_name}")
            self._delete_file_chunks(file_name)
        else:
            print(f"\nIngesting: {file_name}")

        # Load document
        documents = self.load_document(file_path)

        # Chunk documents
        chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)

        # Create/update vector store
        vectorstore = self.create_or_update_vector_store(chunks)

        # Mark as ingested
        self._mark_file_ingested(file_path)

        print(f"Ingestion complete.\n")

        return vectorstore

    def ingest_folder(self, folder_path: str, chunk_size: int = 500, chunk_overlap: int = 50,
                     extensions: Optional[Set[str]] = None, force: bool = False) -> Chroma:
        """
        Ingest all supported files in a folder.
        Only processes new or modified files unless force=True.

        Args:
            folder_path: Path to folder containing documents
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            extensions: Set of file extensions to process (default: all supported)
            force: Force re-ingestion of all files

        Returns:
            Chroma vector store instance
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        extensions = extensions or self.SUPPORTED_EXTENSIONS

        print(f"\nScanning folder: {folder_path}")

        # Find all supported files
        files_to_process = []
        for ext in extensions:
            files_to_process.extend(folder_path.glob(f"*{ext}"))

        if not files_to_process:
            print(f"No supported files found in {folder_path}")
            print(f"   Supported extensions: {extensions}")
            return self.load_existing_vectorstore()

        print(f"Found {len(files_to_process)} file(s): {[f.name for f in files_to_process]}\n")

        # Ingest each file
        ingested_count = 0
        skipped_count = 0

        for file_path in files_to_process:
            result = self.ingest_file(str(file_path), chunk_size, chunk_overlap, force)
            if result is not None:
                ingested_count += 1
            else:
                skipped_count += 1

        print(f"\nFolder ingestion summary:")
        print(f"  Ingested: {ingested_count} file(s)")
        print(f"  Skipped: {skipped_count} file(s) (already processed)\n")

        return self.load_existing_vectorstore()

    def load_existing_vectorstore(self) -> Chroma:
        """
        Load existing ChromaDB vector store from disk.

        Returns:
            Chroma vector store instance

        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run ingestion first."
            )

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        print(f"Loaded existing vector store from {self.persist_directory}")
        return vectorstore

    def get_ingested_files(self) -> Dict[str, str]:
        return self._load_tracking_data()

    def clear_tracking(self) -> None:
        if os.path.exists(self.tracking_file):
            os.remove(self.tracking_file)
            print("Cleared file tracking data")


# Backward compatibility alias
PDFIngestor = DocumentIngestor


def main():
    """Example usage of DocumentIngestor."""
    # Initialize ingestor
    ingestor = DocumentIngestor(persist_directory="./chroma_db")

    # Ingest entire folder
    vectorstore = ingestor.ingest_folder("./data")

    # Show ingested files
    print("\nIngested files:")
    for file_path in ingestor.get_ingested_files():
        print(f"  - {Path(file_path).name}")


if __name__ == "__main__":
    main()
