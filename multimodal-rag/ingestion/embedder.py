# ingestion/embedder.py

import google.generativeai as genai
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ── CLIP setup (local, free) ─────────────────────────────────────
print("Loading CLIP model...")
CLIP_MODEL     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP ready.")


# ── Text Embedding (Gemini) ──────────────────────────────────────
def embed_text(text: str) -> list[float]:
    """
    Embed a single text using Gemini embedding model.
    Returns a 768-dim vector.
    """
    text = text.replace("\n", " ").strip()
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]


def embed_texts_batch(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """
    Embed multiple texts in batches.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [t.replace("\n", " ").strip() for t in batch]

        for text in batch:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            all_embeddings.append(result["embedding"])

        print(f"  Embedded batch {i // batch_size + 1} ({len(batch)} texts)")

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a user query using Gemini.
    Uses retrieval_query task type — different from document embedding.
    This distinction improves retrieval accuracy.
    """
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return result["embedding"]


# ── Image Embedding (CLIP — local) ───────────────────────────────
def embed_image(image_path: str) -> list[float]:
    """
    Embed a single image using CLIP.
    Returns a 512-dim vector.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt")

    with torch.no_grad():
        features = CLIP_MODEL.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().tolist()


def embed_query_for_image(query: str) -> list[float]:
    """
    Embed a text query through CLIP's text encoder.
    Used to find relevant images at search time.
    """
    inputs = CLIP_PROCESSOR(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        features = CLIP_MODEL.get_text_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().tolist()