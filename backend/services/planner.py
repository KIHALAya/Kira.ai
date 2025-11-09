from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini LLM wrapper
from schemas import PaperPlan, SectionPlan
from langchain_core.messages import AIMessage
import json
import re

# -------------------------
# Prompt template
# -------------------------
PLAN_PROMPT = """
You are an academic planner. Given the user prompt and some constraints, return a JSON array of chapters and sections.
User prompt:
{prompt}

Constraints:
- Keep total chapters <= {max_chapters}
- For each chapter provide "chapter", "section", "pages" (estimate), and "notes" (short).
- Return ONLY valid JSON array of objects.

Example output:
[
  {{ "chapter": "Introduction", "section": "Background & Motivation", "pages": 2, "notes":"..." }},
  ...
]
"""

# -------------------------
# Initialize LLM
# -------------------------
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro")  # adjust as needed

# -------------------------
# Planner using RunnableSequence
# -------------------------
def create_plan(prompt: str, max_chapters: int = 5) -> PaperPlan:
    llm_runnable = get_llm()

    # Create a PromptTemplate runnable
    prompt_runnable = PromptTemplate(input_variables=["prompt", "max_chapters"], template=PLAN_PROMPT)

    # Compose RunnableSequence: prompt -> LLM
    chain = prompt_runnable | llm_runnable

    # Run the sequence
    resp = chain.invoke({"prompt": prompt, "max_chapters": max_chapters})

    # -------------------------
    # Parse JSON safely
    # -------------------------


    # resp is the output from chain.invoke()
    if isinstance(resp, AIMessage):
        text = resp.content  # extract string content
    elif isinstance(resp, list):
        # sometimes LCEL returns a list of AIMessage
        text = " ".join([m.content if isinstance(m, AIMessage) else str(m) for m in resp])
    else:
        text = str(resp)

    # Now parse JSON safely
    try:
        arr = json.loads(text)
    except Exception:
        import re
        blocks = re.findall(r'\{.+?\}', text, re.S)
        arr = []
        for b in blocks:
            try:
                obj = json.loads(b)
                arr.append(obj)
            except:
                continue

    # -------------------------
    # Normalize into PaperPlan
    # -------------------------
    sections = []
    for obj in arr:
        sec = SectionPlan(
            chapter=obj.get("chapter", obj.get("title") or "Chapter"),
            section=obj.get("section", obj.get("subtitle") or "Section"),
            pages=obj.get("pages"),
            notes=obj.get("notes")
        )
        sections.append(sec)

    return PaperPlan(title=prompt[:80], total_pages=None, chapters=sections)
