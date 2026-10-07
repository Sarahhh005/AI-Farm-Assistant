
import os
import sys
from pathlib import Path

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from huggingface_hub import InferenceClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser

from rag.schemas import AgriculturalResponse


# ============================================================
# Configuration
# ============================================================

VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"


# ============================================================
# Load FAISS Vector Store
# ============================================================

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


# ============================================================
# Retrieve Relevant Agricultural Context
# ============================================================

def retrieve_context(query, k=3):

    vector_store = load_vector_store()

    # --------------------------------------------------------
    # Normalize query
    # --------------------------------------------------------

    query_normalized = (
        query
        .lower()
        .replace("_", " ")
        .strip()
    )

    # --------------------------------------------------------
    # Get all documents stored in FAISS
    # --------------------------------------------------------

    all_documents = list(
        vector_store.docstore._dict.values()
    )

    exact_matches = []

    # --------------------------------------------------------
    # Search for exact disease match
    # --------------------------------------------------------

    for document in all_documents:

        content = document.page_content

        content_lower = content.lower()

        if "disease:" not in content_lower:
            continue

        # Extract disease section
        disease_part = content_lower.split(
            "disease:",
            1
        )[1]

        if "symptoms:" in disease_part:

            disease_name = disease_part.split(
                "symptoms:",
                1
            )[0].strip()

        else:

            disease_name = disease_part.strip()

        disease_name = (
            disease_name
            .replace("_", " ")
            .strip()
        )

        # ----------------------------------------------------
        # Exact / partial disease matching
        # ----------------------------------------------------

        if (
            query_normalized == disease_name
            or query_normalized in disease_name
            or disease_name in query_normalized
        ):

            exact_matches.append(document)

    # --------------------------------------------------------
    # If exact match exists, prioritize it
    # --------------------------------------------------------

    if exact_matches:

        documents = exact_matches[:k]

        print("\nExact disease match found.")

    # --------------------------------------------------------
    # Otherwise use FAISS semantic similarity
    # --------------------------------------------------------

    else:

        print("\nNo exact disease match found.")
        print("Using FAISS semantic similarity search.")

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": k
            }
        )

        documents = retriever.invoke(query)

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context


# ============================================================
# Generate Structured Agricultural Answer
# ============================================================

def generate_answer(query, context):

    # --------------------------------------------------------
    # Hugging Face Token
    # --------------------------------------------------------

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:

        raise ValueError(
            "HF_TOKEN is not set. "
            "Please set your Hugging Face token first."
        )

    # --------------------------------------------------------
    # Hugging Face Inference Client
    # --------------------------------------------------------

    client = InferenceClient(
        api_key=hf_token,
        provider="auto"
    )

    # --------------------------------------------------------
    # Pydantic Output Parser
    # --------------------------------------------------------

    parser = PydanticOutputParser(
        pydantic_object=AgriculturalResponse
    )

    format_instructions = (
        parser.get_format_instructions()
    )

    # --------------------------------------------------------
    # System Prompt
    # --------------------------------------------------------

    system_prompt = f"""
You are an agricultural assistant.

Answer the user's question using ONLY the
agricultural context provided.

Do not invent agricultural facts.

If the context contains information about
the disease, use that information to fill
the symptoms, management, and prevention fields.

If a field is genuinely missing from the context,
use an empty list for that field.

You MUST return the answer according to the
following format instructions:

{format_instructions}
"""

    # --------------------------------------------------------
    # User Prompt
    # --------------------------------------------------------

    user_prompt = f"""
Agricultural Context:

{context}


User Question:

{query}


Return the answer using the required structured format.
"""

    # --------------------------------------------------------
    # Call LLM
    # --------------------------------------------------------

    completion = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,
        max_tokens=700
    )

    # --------------------------------------------------------
    # Raw LLM Output
    # --------------------------------------------------------

    raw_answer = (
        completion
        .choices[0]
        .message
        .content
    )

    print("\n" + "=" * 60)
    print("RAW LLM OUTPUT")
    print("=" * 60)

    print(raw_answer)

    # --------------------------------------------------------
    # Parse LLM Output
    # --------------------------------------------------------

    try:

        structured_answer = parser.parse(
            raw_answer
        )

    except Exception as e:

        print("\n" + "=" * 60)
        print("PARSING ERROR")
        print("=" * 60)

        print(e)

        raise

    return structured_answer


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI FARM ASSISTANT - STRUCTURED RAG")
    print("=" * 60)

    query = "What are the symptoms of Apple Scab?"

    print("\nUser Question:")
    print(query)

    # --------------------------------------------------------
    # Retrieve Context
    # --------------------------------------------------------

    print("\nSearching FAISS...")

    context = retrieve_context(
        query,
        k=3
    )

    print("\n" + "=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)

    print(context)

    # --------------------------------------------------------
    # Generate Answer
    # --------------------------------------------------------

    print("\nGenerating structured answer...")

    answer = generate_answer(
        query,
        context
    )

    # --------------------------------------------------------
    # Final Structured Answer
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STRUCTURED FINAL ANSWER")
    print("=" * 60)

    print("\nDisease:")
    print(answer.disease)

    print("\nSymptoms:")

    for symptom in answer.symptoms:
        print("-", symptom)

    print("\nManagement:")

    for item in answer.management:
        print("-", item)

    print("\nPrevention:")

    for item in answer.prevention:
        print("-", item)

