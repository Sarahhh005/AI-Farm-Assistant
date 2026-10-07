from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path("rag/documents")


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        docs = loader.load()
        documents.extend(docs)

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    print("=" * 60)
    print("RAG DOCUMENT CHUNKING")
    print("=" * 60)

    documents = load_documents()

    print("Original documents:", len(documents))

    chunks = split_documents(documents)

    print("Total chunks:", len(chunks))

    print("\n" + "=" * 60)
    print("CHUNK DETAILS")
    print("=" * 60)

    for i, chunk in enumerate(chunks):

        print(f"\nChunk {i + 1}")
        print("-" * 40)

        print("Source:", chunk.metadata.get("source"))
        print("Characters:", len(chunk.page_content))

        print("\nContent:")
        print(chunk.page_content[:300])

        print("-" * 40)