# api/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from ingestion import parse_pdf, index_document
from retrieval import retrieve_all, generate_answer

app = FastAPI(
    title="Multimodal RAG API",
    description="Upload a PDF and ask questions across text, tables and images.",
    version="1.0.0"
)

# temp folder for uploaded PDFs
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Health Check ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "Multimodal RAG API is running"}


# ── Upload + Index ────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    Parses and indexes all text, images and tables into ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # save to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # parse + index
    try:
        parsed = parse_pdf(file_path)
        index_document(parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return {
        "message"     : "PDF indexed successfully",
        "filename"    : file.filename,
        "text_chunks" : len(parsed["text_chunks"]),
        "images"      : len(parsed["images"]),
        "tables"      : len(parsed["tables"])
    }


# ── Query ─────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    """
    Ask a question about the uploaded document.
    Returns answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retrieved = retrieve_all(request.question)
        result    = generate_answer(request.question, retrieved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return {
        "question"   : result["query"],
        "answer"     : result["answer"],
        "cited_pages": result["cited_pages"],
        "sources"    : {
            "text_chunks": len(result["sources"]["text"]),
            "tables"     : len(result["sources"]["tables"]),
            "images"     : len(result["sources"]["images"])
        }
    }