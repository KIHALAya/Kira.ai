from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from services.ingestion import process_docs
from services.embedding import embed_chunks
import shutil
from schemas import Document
import os 

app = FastAPI(title="AI Research Paper Agent Backend")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon — open CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health check ---
@app.get("/")
def root():
    return {"message": "Backend is running 🚀"}

# --- Ingestion endpoint ---
@app.post("/ingest/")
async def ingest_data(
    prompt: str = Form(...),
    files: List[UploadFile] = File(None)
):
    try:
        os.makedirs("temp_uploads", exist_ok=True)
        uploaded_files: List[Document] = []

        for file in files:
            file_location = f"temp_uploads/{file.filename}"
            with open(file_location, "wb") as f:
                shutil.copyfileobj(file.file, f)
            uploaded_files.append(Document(name=file.filename, file_path=file_location))

        # Process uploaded files
        docs_data = process_docs(uploaded_files)
        if not docs_data:
            return {"status": "error", "message": "No valid chunks to embed"}
        # Create embeddings
        index_path = embed_chunks(docs_data)

        # Optional: delete temp files
        for doc in uploaded_files:
            if doc.file_path and os.path.exists(doc.file_path):
                os.remove(doc.file_path)

        return {
            "status": "success",
            "message": "Data ingested and embedded successfully",
            "faiss_index_path": index_path,
            "num_chunks": len(docs_data)
    }
    except Exception as e:
        return {"status": "error", "message": str(e)}
