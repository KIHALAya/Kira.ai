# test_pipeline.py
from schemas import Document
from services.ingestion import process_docs
from services.embedding import embed_chunks, load_vectorstore
from agents.paper_agent import PaperAgent

# -------------------------
# 1) Prepare raw documents
# -------------------------
# Provide paths to local PDF, DOCX, or image files
print("let's start")
docs = [
    Document(name="sample_text", file_path="temp_uploads/02 Challenge - Agentic AI for accelerated Research.docx.pdf"),
    Document(name="diagram", file_path="temp_uploads/image.png")
]

user_prompt = "Generate a concise report for this project that I realized for the 24 hours Global MIT Hackathon - 3rd edition using python FastAPI, LangChain, LangGraph, Streamlit and Google AI Studio "

# -------------------------
# 2) Ingest documents
# -------------------------

ingested_chunks = process_docs(docs)
print(f"Ingested {len(ingested_chunks)} chunks:")
for chunk in ingested_chunks[:3]:
    print(f"- Type: {chunk.type} | Preview: {chunk.content[:80]}")
print("Okay that's weird ")

# -------------------------
# 3) Embed chunks and build FAISS vectorstore
# -------------------------
vectorstore_path = embed_chunks(ingested_chunks)
vectorstore = load_vectorstore(vectorstore_path)
print("Embeddings created and stored in FAISS.")

# -------------------------
# 4) Initialize PaperAgent with real retriever
# -------------------------
agent = PaperAgent(vectorstore=vectorstore)

# -------------------------
# 5) Run agentic pipeline
# -------------------------
result = agent.generate(user_prompt=user_prompt, max_chapters=3, refine_iters=1)

# -------------------------
# 6) Inspect output
# -------------------------
print("\n=== PAPER PLAN ===")
for ch in result["plan"].chapters:
    print(f"- Chapter: {ch.chapter} | Section: {ch.section} | Pages: {ch.pages}")

print("\n=== GENERATED SECTIONS ===")
for sec in result["sections"]:
    print(f"\nChapter: {sec.chapter} | Section: {sec.section}")
    print(f"Content preview: {sec.content[:300]}")
    print(f"Sources: {sec.sources}")
    print(f"Images: {sec.images}")
