"""One-command launcher: serves the API and the frontend on the same port.

    python run.py      ->  http://127.0.0.1:8000/app/

Nothing in the existing project is modified; this file only imports `api.main.app`
and mounts the `frontend/` folder on it (plus CORS, so the UI also works from other origins).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # rag_chain.py loads "rag/vector_store" with a relative path
sys.path.insert(0, str(ROOT))

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.main import app

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/app", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")


@app.get("/ui", include_in_schema=False)
def ui():
    return RedirectResponse("/app/")


if __name__ == "__main__":
    print("\nOpen the app:  http://127.0.0.1:8000/app/\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
