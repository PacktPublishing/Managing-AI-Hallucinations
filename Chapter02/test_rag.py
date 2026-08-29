"""
Simple test script for the RAG system
"""

from src.rag import RAGSystem

def test_rag():

    # Initialize
    rag = RAGSystem(persist_directory="./chroma_db", top_k=3)

    # Ingest all files in data folder (only new/modified files)
    print("Ingesting documents from ./data folder...\n")
    rag.ingest_folder("./data")

    # Show ingested files
    ingested = rag.get_ingested_files()
    if ingested:
        print(f"\n Total ingested files: {len(ingested)}")
        from pathlib import Path
        for file_path in ingested:
            print(f"   - {Path(file_path).name}")

    # Test queries
    test_questions = [
        "What is DataMind?",
        "How can I contact support?"
    ]

    for question in test_questions:
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        print('─'*60)

        result = rag.ask(question)

        print(f"\nA: {result['answer']}")
        print(f"\n Sources: {len(result['sources'])} chunks")
        for i, src in enumerate(result['sources'], 1):
            print(f"   [{i}] Page {src['page']}")



if __name__ == "__main__":
    test_rag()
