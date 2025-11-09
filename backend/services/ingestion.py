from typing import List
from schemas import Document, IngestedChunk
from utils import (
    download_file, read_pdf, read_docx, chunk_text,
    extract_text_from_image, is_image
)
import os

def process_docs(docs: List[Document], chunk_size: int = 500, overlap: int = 50) -> List[IngestedChunk]:
    """
    Process text documents and images, returning unified structured chunks for embedding.
    """
    all_chunks: List[IngestedChunk] = []

    for doc in docs:
        file_path = doc.file_path
        if not file_path and doc.url:
            file_path = download_file(doc.url)

        if not file_path and not doc.content:
            continue

        if file_path and is_image(file_path):
            # Handle image files
            text_from_img = extract_text_from_image(file_path)
            all_chunks.append(
                IngestedChunk(
                    type="image",
                    file_path=file_path,
                    content=text_from_img if text_from_img else None,
                    metadata={"source": doc.name}
                )
            )
        else:
            # Handle text docs
            if doc.content:
                text = doc.content
            elif file_path:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".pdf":
                    text = read_pdf(file_path)
                elif ext in [".docx", ".doc"]:
                    text = read_docx(file_path)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
            else:
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in chunks:
                all_chunks.append(
                    IngestedChunk(
                        type="text",
                        content=chunk,
                        metadata={"source": doc.name}
                    )
                )

    return all_chunks
