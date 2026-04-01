
🚀 HackRx PDF-RAG Engine (Rank 88 — Bajaj Finserv HackRx 2025)

A production-focused AI system for extracting insights from PDF documents using OCR, LLM reasoning, and advanced RAG pipelines.

This repo contains the final solution submitted to Bajaj Finserv HackRx 2025 — delivering high-accuracy PDF Question-Answering, token extraction, flight number resolution, and context-aware policy QA.

🏆 Hackathon Overview — Bajaj HackRx 2025

Bajaj HackRx required participants to build a real-time PDF understanding system capable of:

Extracting information from 100+ page PDFs

Handling image-heavy documents with OCR

Supporting multilingual queries (English + Malayalam)

Enforcing strict answer rules (no clause numbers, no hallucination)

Parsing token URLs (plaintext / JSON)

Solving flight number challenge using custom PDF logic

📊 Performance Across HackRx Levels
Level	Task Description	Accuracy	Score
Level 1	Baseline PDF QA (10–20 PDFs)	8%	45
Level 2	Larger dataset + token extraction	29%	250
Level 3	OCR + Multi-language + Context selection	38%	293
Level 4 (Final)	Full reasoning, token solver, flight solver, Malayalam OCR	67%	694
🎉 Final Rank Achieved: Rank 88 (out of 67 thousands of teams)

Your final Level-4 engine demonstrated strong reasoning, efficient context-building, and robust policy extraction.

⭐ Key Features
🔹 1. Advanced PDF Text Extraction

Hybrid extraction using PyMuPDF + OCR (Tesseract)

Malayalam-aware OCR

JPEG/PNG-heavy page fallback

Auto-cleaning + Unicode normalization

🔹 2. Section-Aware Context RAG

Auto-detects document headings

Builds semantic section index

Selects top-relevant sections using similarity scores

🔹 3. Chunk-Level Relevance Ranking

Numerical evidence scoring

Neighbour-chunk expansion

Up to 20 context slices merged intelligently

🔹 4. LLM Pipeline (Mistral Primary + Groq Fallback)

Uses Mistral-Small for deterministic context QA

Uses Groq (Llama 4 Scout 17B) as automatic fallback

Automatic language rewriting (Malayalam ↔ English)

🔹 5. Mission Token Extraction

Detects tokens in:

raw text

HTML

JSON

<code>/<pre> tags

nested payloads

Handles 16–256 character secrets

🔹 6. Ticket Flight Number Solver

Dynamically parses:

City → Landmark tables

Landmark → Endpoint mapping rules

Zero hardcoded cities

Hits the correct flight endpoint

Robust to multi-word names (e.g., “Dubai Airport”)

🔹 7. Optimized FastAPI Serving

HTTP/2 async client

Reduced timeouts

Retry logic

Section-aware answer shaping

🗂️ Folder Structure
Bajaj_json/
│
├── main.py                 # Full FastAPI backend (Level 4 engine)
├── requirements.txt        # Python dependencies
├── .env                    # API_TOKEN, MISTRAL_API_KEY, GROQ_API_KEY
├── .GITIGNORE              # Ignore cache, env, temp files
│
├── readme.md               # Project documentation (this file)
│
├── build.sh                # Build script for deployment (Render)
├── start.sh                # Start script for Render
├── render.yaml             # Render deployment config
│
└── Dataset/                # (Optional) Local working files

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/deveshh01/HackRx-6.O.git



2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variables (.env)

Create .env file:

API_TOKEN=your_api_token_here
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key

▶️ Run the FastAPI Server
uvicorn main:app --host 0.0.0.0 --port 8000


Check server:

http://localhost:8000/

🔌 API Endpoint
POST /api/v1/hackrx/run
Request Body
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the waiting period?",
    "Is accidental death covered?"
  ]
}

Headers
Authorization: Bearer <API_TOKEN>
Content-Type: application/json

Response
{
  "answers": [
    "The waiting period is 24 months...",
    "Yes. It is covered as per..."
  ]
}

🧠 Architecture Overview

                ┌──────────────────────────┐
                │         CLIENT            │
                │  (Postman / Portal)       │
                └────────────┬─────────────┘
                             │
                ┌────────────▼──────────────┐
                │        FastAPI             │
                │  /api/v1/hackrx/run        │
                └────────────┬──────────────┘
                             │
     ┌───────────────────────┼────────────────────────┐
     │                       │                        │
┌────▼─────┐         ┌──────▼──────┐         ┌───────▼────────┐
│ PDF Text │         │ Section RAG  │         │ Token/Flight   │
│ Extract  │         │ Builder      │         │ Solvers        │
└────▲─────┘         └──────▲──────┘         └───────▲────────┘
     │                       │                        │
     └──────────────┬────────┴──────────────┬────────┘
                    │                       │
           ┌────────▼────────┐     ┌────────▼────────┐
           │ Mistral LLM      │     │ Groq LLM        │
           │ (Primary Engine) │     │ (Fallback)       │
           └──────────────────┘     └──────────────────┘


🚀 Why This System Scored Highest in Level-4

✔ Malayalam-aware OCR
✔ Relevance-driven chunking
✔ Section detection
✔ Token solver
✔ Ticket → Landmark → Endpoint → Flight solver
✔ Zero hallucination rules
✔ Efficient runtime (~3–6 seconds/query)
✔ Clean reasoning pipeline

This matches real enterprise PDF QA behavior.

🔮 Future Enhancements
1️⃣ Integrate FAISS Vector RAG

Store section embeddings to speed up context retrieval.

2️⃣ GPU-optimized OCR

Use EasyOCR or PaddleOCR for 3× faster decoding.

3️⃣ Resume-Aware Context

Detect policy domain type automatically (Health / Motor / Loan).

4️⃣ Streaming Answers

Return partial responses using FastAPI Server-Sent Events.

5️⃣ Multi-Document QA

Allow batching of 5–10 PDFs in one request.

📝 Final Note

This repository represents the full production engine that achieved:

🥇 Rank 88 in Bajaj Finserv HackRx 2025

With performance peaking at:

Level 4 Accuracy: 67%

Total Score: 694

Strong reasoning + robust extraction + fast runtime


