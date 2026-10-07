# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | *(Sarah Mahmoud fathy)*                   |
| Project Name     | AI Farm Assistant                    |
| GitHub Username  | *(https://github.com/Sarahhh005)*             |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

**AI Farm Assistant** is an end-to-end system that helps farmers diagnose plant diseases from a single leaf photo and get practical, grounded guidance.

The user uploads a photo of a leaf. A fine-tuned computer-vision model identifies the crop and disease, then a **Retrieval-Augmented Generation (RAG)** pipeline looks up that disease in an agricultural knowledge base and an LLM turns the retrieved facts into a structured answer: **symptoms, management, and prevention**.

Because the LLM is instructed to answer *only* from the retrieved context, the advice stays tied to the knowledge base instead of being invented.

```
Leaf photo ──► EfficientNet-B0 ──► disease + confidence
                                        │
                                        ▼
                      FAISS retrieval (top-3 chunks)
                                        │
                                        ▼
              LLM (structured output via Pydantic parser)
                                        │
                                        ▼
            FastAPI  /predict  ──►  Web frontend (EN / AR)
```

---

# ✨ Features

* 🌿 **Disease detection** for 38 classes (14 crops, healthy and diseased) using a fine-tuned EfficientNet-B0.
* 📚 **RAG knowledge base** covering every diseased class plus healthy plants, split into self-contained chunks so retrieval always carries the crop and disease name.
* 🧠 **Structured LLM output** (disease, symptoms, management, prevention) enforced with a Pydantic output parser.
* ⚡ **REST API** built with FastAPI (`/`, `/health`, `/predict`).
* 🖥️ **Modern web frontend**: drag-and-drop upload, image preview, progress steps, confidence ring, low-confidence warning, light/dark theme, and an **English / Arabic (RTL)** interface.
* 🚀 **One-command launcher** (`run.py`) that serves the API and the frontend on a single port.

---

# 🛠️ Technologies Used

| Area              | Tools                                                                  |
| ----------------- | ---------------------------------------------------------------------- |
| Computer Vision   | PyTorch, torchvision (EfficientNet-B0), Pillow                         |
| RAG               | LangChain, FAISS (`faiss-cpu`), `sentence-transformers/all-MiniLM-L6-v2` |
| LLM               | Hugging Face Inference API (`openai/gpt-oss-120b`), Pydantic           |
| Backend           | FastAPI, Uvicorn, python-multipart                                     |
| Frontend          | HTML, CSS, vanilla JavaScript (single file, no dependencies)           |
| Language          | Python 3.11                                                            |

### Project structure

```
AI-Farm-Assistant/
├── api/                    # FastAPI app (main.py)
├── computer_vision/
│   └── inference/          # predictor.py – loads the model and classifies an image
├── integration/            # pipeline.py – CV → RAG → LLM
├── rag/
│   ├── documents/          # knowledge base (.txt files)
│   ├── ingestion/          # loading and chunking scripts
│   ├── embeddings/         # create_vector_store.py
│   ├── retrieval/          # rag_chain.py – retrieval + LLM generation
│   ├── vector_store/       # FAISS index
│   └── schemas.py          # Pydantic response schema
├── models/                 # EfficientNet-B0 weights + class mapping + config
├── frontend/               # index.html – web UI
├── run.py                  # launches API + frontend together
└── requirements.txt
```

### How it works

1. **Vision** – `predictor.py` resizes the image to 224×224, normalizes it with ImageNet statistics, and runs EfficientNet-B0 (fine-tuned on 38 PlantVillage-style classes) to get the disease label and softmax confidence.
2. **Normalization** – a label such as `Tomato___Early_blight` becomes `Tomato Early_blight` and is used as the retrieval query.
3. **Retrieval** – the query is embedded with `all-MiniLM-L6-v2` and the top 3 chunks are fetched from the FAISS index.
4. **Generation** – the chunks and the query are sent to the LLM with strict instructions to use only the provided context and to return JSON matching the `AgriculturalResponse` schema.
5. **Serving** – FastAPI returns the disease, confidence, and answer; the frontend renders them.

### Knowledge base

The knowledge base lives in `rag/documents/` and is made of plain-text records in a fixed format:

```
Crop: Tomato | Disease: Leaf Mold (Tomato___Leaf_Mold)
Symptoms: ...
Management: ...
Prevention: ...
```

* `plant_diseases.txt` – the original records (Apple Scab, Apple Black Rot, Tomato Early/Late Blight, Potato Early Blight).
* `plant_diseases_extended.txt` – the remaining diseased classes and a record for healthy plants.

Each record is kept under the 500-character chunk size, so it stays in one chunk and always keeps its crop and disease name. Management advice is intentionally general ("per local recommendations") and contains no pesticide doses.

---

# ⚙️ Installation

**Requirements:** Python 3.11 and a free [Hugging Face access token](https://huggingface.co/settings/tokens).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AI-Farm-Assistant.git
cd AI-Farm-Assistant

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build the vector store (run again whenever you edit rag/documents/*.txt)
python rag/embeddings/create_vector_store.py

# 5. Set your Hugging Face token
set HF_TOKEN=your_token_here           # Windows (cmd)
# $env:HF_TOKEN="your_token_here"      # Windows (PowerShell)
# export HF_TOKEN=your_token_here      # Linux / macOS
```

> ⚠️ Run all commands from the **project root**, because the RAG code loads `rag/vector_store` with a relative path.

---

# 🚀 Usage

### Option 1: Web app (recommended)

```bash
python run.py
```

Open **http://127.0.0.1:8000/app/**, upload a leaf photo, and press **Analyze plant**. Use the button in the top bar to switch between English and Arabic.

### Option 2: API only

```bash
uvicorn api.main:app --reload
```

Interactive docs are available at `http://127.0.0.1:8000/docs`.

| Method | Endpoint   | Description                                  |
| ------ | ---------- | -------------------------------------------- |
| GET    | `/`        | Service status                               |
| GET    | `/health`  | Health check                                 |
| POST   | `/predict` | Upload an image (`file`) and get a diagnosis |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@test_image.jpg"
```

Response shape:

```json
{
  "success": true,
  "disease": "Tomato Early_blight",
  "confidence": 97.4,
  "answer": "disease='...' symptoms=[...] management=[...] prevention=[...]"
}
```

### Option 3: Command line

```bash
python integration/pipeline.py      # runs the full pipeline on test_image.jpg
python rag/retrieval/rag_chain.py   # tests RAG on its own
```

---

# 📸 Demo

*Add screenshots or a short screen recording here, for example:*

<img width="1497" height="923" alt="Screenshot 2026-10-07 043331" src="https://github.com/user-attachments/assets/6f7bce53-969f-453b-b045-641789aebea2" />


---

# 📈 Results

* Built a complete working pipeline that connects computer vision, retrieval, and an LLM behind one API and one web interface.
* Fine-tuned EfficientNet-B0 on **38 classes** (250 images per class, fixed seed `42` for reproducibility).
* Produced a knowledge base with coverage for **all 26 disease classes** plus healthy plants.
* Delivered structured, schema-validated answers (symptoms / management / prevention) instead of free text.
* Shipped a bilingual (EN/AR) frontend that works with the API without any change to the backend code.

*Add your model's validation accuracy here once measured, for example: "Validation accuracy: XX.X%".*

---

# 🔮 Future Improvements

* Return the answer as real JSON from the API instead of a stringified object.
* Generate answers in Arabic, and expand the knowledge base with more crops and local conditions.
* Add retrieval filtering by crop and disease metadata to improve precision.
* Add a "healthy / not a leaf" rejection step for low-quality or irrelevant images.
* Add a follow-up chat so users can ask questions about the diagnosis.
* Containerize with Docker and add automated tests.

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**, and it will be showcased on the official [Tips Hindawi](https://www.tipshindawi.com/) website.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
