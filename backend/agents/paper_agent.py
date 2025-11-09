# agents/paper_agent.py
from typing import List
from services.planner import create_plan
from services.generator import generate_section
from schemas import PaperPlan, GeneratedSection
from langchain_core.documents import Document as LC_Document

# Optional search util (SerpAPI). If SERPAPI_API_KEY not set, this will be skipped.
from langchain_community.utilities import SerpAPIWrapper

class PaperAgent:
    def __init__(self, vectorstore=None, search_api_key: str = None):
        """
        vectorstore: FAISS vectorstore instance (LangChain FAISS)
        """
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 6}) if vectorstore else None
        self.search = SerpAPIWrapper() if (search_api_key) else None

    def _fact_check(self, claim: str) -> List[str]:
        """Quick web check using search wrapper. Returns list of URLs/snippets"""
        if not self.search:
            return []
        results = self.search.run(claim, num_results=5)
        # SerpAPIWrapper returns a string — best-effort parse
        return [results]

    def generate(self, user_prompt: str, max_chapters: int = 5, refine_iters: int = 1):
        # 1. Create plan
        plan = create_plan(user_prompt, max_chapters=max_chapters)

        generated: List[GeneratedSection] = []

        # 2. Iterate chapters/sections
        for sec in plan.chapters:
            chapter = sec.chapter
            section = sec.section
            pages = sec.pages or 1
            notes = sec.notes or ""

            # Generate initial draft
            out = generate_section(
                user_prompt=user_prompt,
                chapter=chapter,
                section=section,
                pages=pages,
                retriever=self.retriever,
                notes=notes
            )

            content = out["content"]
            sources = out["sources"]
            images = out["images"]

            # 3. Optional fact-check & refinement loop (CoT)
            for i in range(refine_iters):
                # basic check: for each top claim (could parse with regex), call fact-check
                # For MVP, we call fact-check on the whole section title and summary.
                if self.search:
                    fc = self._fact_check(f"{chapter} {section} {user_prompt}")
                else:
                    fc = []

                # Ask LLM to refine given the fact-check and current content - use generator again with context
                # For speed, do one more generation pass with the fact-check appended to context via retriever hack:
                # Here we emulate by updating notes
                notes_with_fc = notes + "\n\nFactCheckResults:\n" + ("\n".join(fc)[:2000] if fc else "none")
                out_refined = generate_section(
                    user_prompt=user_prompt,
                    chapter=chapter,
                    section=section,
                    pages=pages,
                    retriever=self.retriever,
                    notes=notes_with_fc
                )
                # adopt refined content if different
                content = out_refined["content"]
                sources = list(set(sources + out_refined.get("sources", [])))
                images = images + out_refined.get("images", [])

            generated.append(GeneratedSection(
                chapter=chapter,
                section=section,
                content=content,
                sources=sources,
                images=images
            ))

        # final package
        return {
            "plan": plan,
            "sections": generated
        }
