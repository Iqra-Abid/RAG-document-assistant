from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


def main():

    # Load documents
    docs = load_all_documents("data")


    # Initialize vector store
    store = FaissVectorStore(
        persist_dir="faiss_store"
    )


    # Load existing index or build new one
    if store.exists():

        print("[INFO] Existing FAISS index found")
        store.load()

    else:

        print("[INFO] No FAISS index found. Creating...")
        store.build_from_documents(
            docs
        )


    # Initialize RAG search
    rag_search = RAGSearch(
        vector_store=store
    )


    query = "python"


    result = rag_search.search_and_summarize(
        query,
        top_k=3
    )


    print("\nSummary:")
    print(result)



if __name__ == "__main__":
    main()