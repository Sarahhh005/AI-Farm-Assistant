from pathlib import Path

from langchain_community.document_loaders import TextLoader


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


if __name__ == "__main__":

    documents = load_documents()

    print("=" * 60)
    print("DOCUMENT LOADING")
    print("=" * 60)

    print("Number of documents:", len(documents))

    for document in documents:

        print("\nSource:", document.metadata)
        print("Characters:", len(document.page_content))
        print("-" * 60)