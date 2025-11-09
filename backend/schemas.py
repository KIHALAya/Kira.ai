from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union


class SectionPlan(BaseModel):
    chapter: str
    section: str
    pages: Optional[int] = None
    notes: Optional[str] = None  # planner hints

class PaperPlan(BaseModel):
    title: str
    total_pages: Optional[int] = None
    chapters: List[SectionPlan]

class GeneratedSection(BaseModel):
    chapter: str
    section: str
    content: str
    sources: List[str] = []
    images: List[dict] = []  # {file_path or url, caption, metadata}


class Document(BaseModel):
    name: str
    url: Optional[HttpUrl] = None
    content: Optional[str] = None
    file_path: Optional[str] = None

class IngestedChunk(BaseModel):
    type: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[dict] = {}

class IngestionRequest(BaseModel):
    prompt: str
    docs: List[Document]
    images: Optional[List[Document]] = []
