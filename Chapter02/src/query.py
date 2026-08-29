"""
Query Module for RAG System
Handles question answering using ChromaDB retrieval and OpenAI chat completion.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Add project root to path when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


class RAGQueryEngine:
    """Handles question answering using RAG pipeline."""

    def __init__(self, persist_directory: str = "./chroma_db", top_k: int = 3):
        """
        Initialize the RAG query engine.

        Args:
            persist_directory: Directory where ChromaDB is persisted
            top_k: Number of relevant chunks to retrieve
        """
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.persist_directory = persist_directory
        self.top_k = top_k

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.api_key
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0,
            openai_api_key=self.api_key
        )

        # Load vector store
        self.vectorstore = self._load_vectorstore()

        # Create QA chain
        self.qa_chain = self._create_qa_chain()

    def _load_vectorstore(self) -> Chroma:
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run ingestion first using ingest.py"
            )

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        return vectorstore

    def _create_qa_chain(self) -> RetrievalQA:
        # Custom prompt template
        prompt_template = """You are a helpful AI assistant answering questions based on the provided context.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know - don't try to make up an answer.

When answering:
1. Be concise and direct
2. Cite the page number(s) from the source when relevant
3. If the context doesn't contain the answer, clearly state that

Context:
{context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        return qa_chain

    def query(self, question: str, verbose: bool = False) -> Dict[str, any]:
        """
        Query the RAG system with a question.

        Args:
            question: User's question
            verbose: If True, print retrieved context

        Returns:
            Dictionary containing:
                - answer: Generated answer
                - sources: List of source documents with metadata
                - question: Original question
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Run query
        result = self.qa_chain.invoke({"query": question})

        # Extract answer and sources
        answer = result["result"]
        source_docs = result["source_documents"]

        # Format sources with page numbers
        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "N/A")
            })

        # Print verbose output if requested
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}\n")

            print("Retrieved Context:")
            print("-" * 60)
            for i, source in enumerate(sources, 1):
                print(f"\n[Chunk {i}] Page {source['page']}:")
                print(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])

            print(f"\n{'='*60}")
            print(f"Answer:")
            print(f"{'='*60}\n")
            print(answer)
            print()

        return {
            "answer": answer,
            "sources": sources,
            "question": question
        }

    def get_relevant_chunks(self, question: str, k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant chunks without generating an answer.

        Args:
            question: User's question
            k: Number of chunks to retrieve (defaults to self.top_k)

        Returns:
            List of dictionaries containing chunk content and metadata
        """
        k = k or self.top_k
        docs = self.vectorstore.similarity_search(question, k=k)

        chunks = []
        for doc in docs:
            chunks.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "N/A")
            })

        return chunks


def main():
    """Example usage of RAGQueryEngine."""
    # Initialize query engine
    print("Initializing RAG Query Engine...")
    engine = RAGQueryEngine(persist_directory="./chroma_db", top_k=3)
    print("Query engine ready!\n")

    # Example questions
    questions = [
        "What is DataMind?",
        "How can I contact support?",
        "What are the main features?"
    ]

    for question in questions:
        result = engine.query(question, verbose=True)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
