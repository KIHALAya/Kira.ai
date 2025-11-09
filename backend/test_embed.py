import os
from services.ingestion import process_docs
from services.embedding import embed_chunks
from schemas import Document

def test_local_embedding():
    # pick a small file that actually exists
    test_file = "C:\\Users\\HP\\Desktop\\Kira.ai\\backend\\temp_uploads\\Projet_libre_2.pdf"  # ← replace with any file in your project
    if not os.path.exists(test_file):
        print(f"❌ File not found: {test_file}")
        return

    # Step 1: Wrap it in the schema model
    docs = [Document(name=os.path.basename(test_file), file_path=test_file)]

    # Step 2: Run ingestion
    print("🔹 Running process_docs() ...")
    chunks = process_docs(docs)
    print(f"✅ Returned {len(chunks)} chunks")

    # Step 3: Run embedding
    print("🔹 Running embed_chunks() ...")
    index_path = embed_chunks(chunks)
    print(f"✅ Embeddings saved at: {index_path}")

    # Step 4: Confirm FAISS files
    if os.path.exists(index_path):
        print("🎉 FAISS index successfully created.")
    else:
        print("⚠️ FAISS index file not found!")

if __name__ == "__main__":
    test_local_embedding()
