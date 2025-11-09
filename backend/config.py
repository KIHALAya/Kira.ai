import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")

#Vector store config
FAISS_INDEX_PATH = "vectorstores/paper_embeddings.index"
EMBEDDING_DIM = 768