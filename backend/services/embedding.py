import os
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS
from schemas import IngestedChunk
from config import GOOGLE_API_KEY, FAISS_INDEX_PATH
from langchain_core.documents import Document as LC_Document
from pathlib import Path


def get_embedding_model():
    if not GOOGLE_API_KEY:
        raise ValueError("API KEY not found")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)


# ---- Conversion utility ----
def chunks_to_lc_docs(chunks: List[IngestedChunk]) -> List[LC_Document]:
    """Convert ingested chunks to LangChain Documents with metadata."""
    docs = []
    for chunk in chunks:
        if not chunk.content:
            continue
        metadata = chunk.metadata or {}
        metadata["type"] = chunk.type
        metadata["path"] = chunk.file_path
        docs.append(LC_Document(page_content=chunk.content, metadata=metadata))
    return docs

# ---- Embedding workflow ----
def embed_chunks(chunks: List[IngestedChunk], save_dir: str = "faiss_indexes"):
    import uuid
    from pathlib import Path

    if not chunks:
        raise ValueError("No chunks to embed")  # <- fail early

    unique_id = str(uuid.uuid4())
    save_path = Path(save_dir) / f"{unique_id}.faiss"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()
    lc_docs = chunks_to_lc_docs(chunks)
    
    if not lc_docs:
        raise ValueError("No valid documents to embed")  # <- fail early

    vectorstore = FAISS.from_documents(lc_docs, embedding_model)
    vectorstore.save_local(save_path)

    return str(save_path)


def load_vectorstore(save_path: str = FAISS_INDEX_PATH):
    """
    Load an existing FAISS vector store.
    """
    embedding_model = get_embedding_model()
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
