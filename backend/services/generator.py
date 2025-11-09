from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document as LC_Document
from langchain_core.messages import AIMessage
import asyncio
import re
import json

GEN_PROMPT = """
You are an academic writer. Using the user prompt and the retrieved context, draft the content for the SECTION below.
Query: {query}
User prompt: {user_prompt}
SECTION: {chapter} — {section}
Planner notes: {notes}

Context (retrieved relevant passages):
{context}

Instructions:
- Write clear academic prose appropriate for a research paper.
- Include inline citations in square brackets where you reference a source, and return a list of sources below.
- Keep text length suitable for ~{pages} pages (estimate). Be concise but complete.

Return:
1) The content text, and then
2) A JSON object with "sources": [list_of_source_identifiers], "image_suggestions": ["query": "...", "caption": "..."]
"""

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro")

def generate_section(
    user_prompt: str,
    chapter: str,
    section: str,
    pages: Optional[int],
    retriever,  # FAISS retriever
    notes: Optional[str] = ""
):
    llm = get_llm()
    prompt_runnable = PromptTemplate(
        input_variables=["user_prompt", "chapter", "section", "notes", "context", "pages", "query"],
        template=GEN_PROMPT
    )

    # Retrieve context from vectorstore
    query = f"{chapter} {section} {user_prompt}"
    context_docs: List[LC_Document] = retriever._get_relevant_documents(query, run_manager=None)
    context_text = "\n\n".join([
        f"SOURCE:{d.metadata.get('path') or d.metadata.get('source','unknown')}\n{d.page_content}"
        for d in context_docs[:6]
    ])

    # Compose chain using pipe operator
    chain = prompt_runnable | llm

    # Run chain
    raw = chain.invoke({
        "user_prompt": user_prompt,
        "chapter": chapter,
        "section": section,
        "notes": notes or "",
        "context": context_text,
        "pages": pages or 1,
         "query": f"{chapter} {section} {user_prompt}"
    })

    # -------------------------
    # Extract text from AIMessage
    # -------------------------

    if isinstance(raw, AIMessage):
        text = raw.content
    elif isinstance(raw, list):
        text = " ".join([m.content if isinstance(m, AIMessage) else str(m) for m in raw])
    else:
        text = str(raw)

    # -------------------------
    # Extract content + JSON metadata
    # -------------------------
    json_obj = {}
    m = re.search(r'(\{[\s\S]*\})\s*$', text)
    if m:
        try:
            json_obj = json.loads(m.group(1))
            content = text[:m.start()].strip()
        except Exception:
            content = text
    else:
        content = text

    sources = json_obj.get("sources", [])
    images = json_obj.get("image_suggestions", [])

    return {
        "content": content,
        "sources": sources,
        "images": images,
        "raw": text
    }
