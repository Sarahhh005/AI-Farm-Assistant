import sys
from pathlib import Path

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from computer_vision.inference.predictor import predict
from rag.retrieval.rag_chain import retrieve_context, generate_answer


# ============================================================
# IMAGE
# ============================================================

IMAGE_PATH = PROJECT_ROOT / "test_image.jpg"


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(image_path):

    print("=" * 70)
    print("AI FARM ASSISTANT")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: COMPUTER VISION
    # --------------------------------------------------------

    prediction = predict(image_path)

    disease_raw = prediction["disease"]
    confidence = prediction["confidence_percent"]

    print("\n[1] COMPUTER VISION")
    print("-" * 70)
    print(f"Disease: {disease_raw}")
    print(f"Confidence: {confidence:.2f}%")

    # --------------------------------------------------------
    # STEP 2: NORMALIZE DISEASE NAME
    # --------------------------------------------------------

    disease_name = disease_raw.replace("___", " ")

    print("\n[2] NORMALIZED DISEASE")
    print("-" * 70)
    print(f"Disease: {disease_name}")

    # --------------------------------------------------------
    # STEP 3: RETRIEVE INFORMATION FROM FAISS
    # --------------------------------------------------------

    print("\n[3] RAG RETRIEVAL")
    print("-" * 70)
    print("Searching agricultural knowledge base...")

    context = retrieve_context(disease_name)

    print("Relevant information retrieved successfully.")

    # --------------------------------------------------------
    # STEP 4: GENERATE ANSWER USING LLM
    # --------------------------------------------------------

    print("\n[4] LLM GENERATION")
    print("-" * 70)
    print("Generating agricultural response...")

    answer = generate_answer(disease_name, context)

    # --------------------------------------------------------
    # STEP 5: FINAL RESULT
    # --------------------------------------------------------

    print("\n[5] FINAL RESULT")
    print("=" * 70)

    print(f"\nDisease: {disease_name}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nAgricultural Recommendation:")
    print(answer)

    print("\n" + "=" * 70)

    return {
        "disease": disease_name,
        "confidence": confidence,
        "answer": answer,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_pipeline(IMAGE_PATH)