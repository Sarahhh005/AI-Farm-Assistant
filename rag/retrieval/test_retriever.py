from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTOR_STORE_DIR = Path("rag/vector_store")


def load_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


if __name__ == "__main__":

    print("=" * 60)
    print("RAG RETRIEVER TEST")
    print("=" * 60)

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    query = "What are the symptoms of Apple Scab?"

    print("\nQuery:")
    print(query)

    print("\nSearching FAISS...")

    results = retriever.invoke(query)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, document in enumerate(results):

        print(f"\nResult {i + 1}")
        print("-" * 60)

        print("Source:")
        print(document.metadata.get("source"))

        print("\nContent:")
        print(document.page_content)

        print("-" * 60)