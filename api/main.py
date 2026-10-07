
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integration.pipeline import run_pipeline


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="AI Farm Assistant API",
    description="Plant disease detection and agricultural recommendation API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AI Farm Assistant API is running",
        "status": "healthy"
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# Plant Disease Prediction
# ============================================================

@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate uploaded file
    # --------------------------------------------------------

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # --------------------------------------------------------
    # Temporary image directory
    # --------------------------------------------------------

    temp_dir = PROJECT_ROOT / "api" / "temp"
    temp_dir.mkdir(exist_ok=True)

    image_path = temp_dir / file.filename

    try:

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        contents = await file.read()

        with open(image_path, "wb") as f:
            f.write(contents)

        # ----------------------------------------------------
        # Run complete AI pipeline
        # ----------------------------------------------------

        result = run_pipeline(image_path)

        # ----------------------------------------------------
        # Structured LLM answer
        # ----------------------------------------------------

        answer = result["answer"]

        # ----------------------------------------------------
        # Return JSON response
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "success": True,
                "disease": result["disease"],
                "confidence": result["confidence"],
                "answer": {
                    "disease": answer.disease,
                    "symptoms": answer.symptoms,
                    "management": answer.management,
                    "prevention": answer.prevention
                }
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary image
        # ----------------------------------------------------

        if image_path.exists():
            image_path.unlink()

