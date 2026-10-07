from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DOCUMENTS_DIR = Path("rag/documents")
VECTOR_STORE_DIR = Path("rag/vector_store")


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTOR_STORE_DIR)
    )

    return vector_store


if __name__ == "__main__":

    print("=" * 60)
    print("CREATING FAISS VECTOR STORE")
    print("=" * 60)

    documents = load_documents()

    print("Documents:", len(documents))

    chunks = split_documents(documents)

    print("Chunks:", len(chunks))

    vector_store = create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)

    print("FAISS vector store saved to:")
    print(VECTOR_STORE_DIR)