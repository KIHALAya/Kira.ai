import os
import requests
from  pathlib import Path
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from typing import List
from PIL import Image
import pytesseract

def download_file(url: str, save_dir = "downloads") -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    local_path = os.path.join(save_dir, url.split("/")[-1])
    r = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path


def is_image(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def read_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(file_path: str) -> str:
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR failed for {file_path}: {e}")
    
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50)-> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks