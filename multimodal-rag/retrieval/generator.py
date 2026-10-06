# retrieval/generator.py

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = genai.GenerativeModel("gemini-3.8-flash")

def build_prompt(query: str, retrieved: dict) -> str:
    """
    Build a structured prompt from retrieved context.
    Clearly separates text, table, and image references.
    """
    prompt = f"""You are a helpful assistant answering questions about a document.
Answer ONLY based on the context provided below.
If the context does not contain enough information, say "I couldn't find that in the document."
Always end your answer with a citation in this format: [Source: Page X]

USER QUESTION:
{query}

"""

    # ── Text context ─────────────────────────────────────────────
    if retrieved["text"]:
        prompt += "── TEXT CONTEXT ──\n"
        for chunk in retrieved["text"]:
            prompt += f"[Page {chunk['page']}] {chunk['content']}\n\n"

    # ── Table context ─────────────────────────────────────────────
    if retrieved["tables"]:
        prompt += "── TABLE CONTEXT ──\n"
        for table in retrieved["tables"]:
            prompt += f"[Page {table['page']}]\n{table['content']}\n\n"

    # ── Image references ──────────────────────────────────────────
    if retrieved["images"]:
        prompt += "── RELEVANT IMAGES FOUND ──\n"
        for img in retrieved["images"]:
            prompt += f"[Page {img['page']}] Image file: {img['path']}\n"
        prompt += "\nIf the question relates to a visual, mention which page the image is on.\n"

    return prompt


def generate_answer(query: str, retrieved: dict) -> dict:
    """
    Generate an answer using Gemini with retrieved context.
    Returns answer text + citation metadata.
    """
    prompt = build_prompt(query, retrieved)

    response = MODEL.generate_content(prompt)
    answer   = response.text.strip()

    # collect all cited pages
    cited_pages = sorted(set(
        [c["page"] for c in retrieved["text"]] +
        [t["page"] for t in retrieved["tables"]] +
        [i["page"] for i in retrieved["images"]]
    ))

    return {
        "query"      : query,
        "answer"     : answer,
        "cited_pages": cited_pages,
        "sources"    : {
            "text"  : retrieved["text"],
            "tables": retrieved["tables"],
            "images": retrieved["images"]
        }
    }