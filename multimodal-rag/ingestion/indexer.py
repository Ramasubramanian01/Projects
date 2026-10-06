# ingestion/indexer.py

import chromadb
# from chromadb.config import Settings

# from ingestion.embedder import embed_texts_batch, embed_image
from ingestion.embedder import embed_texts_batch, embed_image

# ── ChromaDB client (local, no server needed) ────────────────────
CLIENT = chromadb.PersistentClient(path="./chroma_store")
# CLIENT = chromadb.PersistentClient(
#     path="./chroma_store",
#     settings=Settings(anonymized_telemetry=False),
# )

# # separate collections for each modality
# TEXT_COLLECTION  = CLIENT.get_or_create_collection(name="text_chunks")
# IMAGE_COLLECTION = CLIENT.get_or_create_collection(name="image_chunks")
# TABLE_COLLECTION = CLIENT.get_or_create_collection(name="table_chunks")
# ingestion/indexer.py  — update collection creation only

TEXT_COLLECTION = CLIENT.get_or_create_collection(
    name="text_chunks",
    metadata={"hnsw:space": "cosine"}   # cosine works best for Gemini
)

TABLE_COLLECTION = CLIENT.get_or_create_collection(
    name="table_chunks",
    metadata={"hnsw:space": "cosine"}
)

# IMAGE_COLLECTION stays the same — CLIP is still 512-dim
IMAGE_COLLECTION = CLIENT.get_or_create_collection(
    name="image_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ── Index Text ───────────────────────────────────────────────────
def index_text_chunks(chunks: list[dict]):
    """
    Embed and store all text chunks in ChromaDB.
    """
    if not chunks:
        print("No text chunks to index.")
        return

    print(f"Indexing {len(chunks)} text chunks...")
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts_batch(texts)

    TEXT_COLLECTION.add(
        ids        = [f"text_{c['chunk_index']}" for c in chunks],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [{"page": c["page"], "type": "text"} for c in chunks]
    )
    print(f"  ✓ {len(chunks)} text chunks indexed")


# ── Index Images ─────────────────────────────────────────────────
def index_images(images: list[dict]):
    """
    Embed and store all images in ChromaDB.
    """
    if not images:
        print("No images to index.")
        return

    print(f"Indexing {len(images)} images...")

    for img in images:
        try:
            embedding = embed_image(img["path"])
            IMAGE_COLLECTION.add(
                ids        = [f"img_p{img['page']}_i{img['image_index']}"],
                embeddings = [embedding],
                documents  = [img["path"]],
                metadatas  = [{"page": img["page"], "type": "image", "path": img["path"]}]
            )
            print(f"  ✓ Indexed image: {img['path']}")
        except Exception as e:
            print(f"  ✗ Failed image {img['path']}: {e}")


# ── Index Tables ─────────────────────────────────────────────────
def index_tables(tables: list[dict]):
    """
    Embed and store all tables in ChromaDB.
    Tables are treated like text — embedded with ada-002.
    """
    if not tables:
        print("No tables to index.")
        return

    print(f"Indexing {len(tables)} tables...")
    texts = [t["content"] for t in tables]
    embeddings = embed_texts_batch(texts)

    TABLE_COLLECTION.add(
        ids        = [f"table_p{t['page']}_i{t['table_index']}" for t in tables],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [{"page": t["page"], "type": "table"} for t in tables]
    )
    print(f"  ✓ {len(tables)} tables indexed")


# ── Master Indexer ───────────────────────────────────────────────
def index_document(parsed: dict):
    """
    Takes the output of parse_pdf() and indexes everything.
    """
    index_text_chunks(parsed["text_chunks"])
    index_images(parsed["images"])
    index_tables(parsed["tables"])
    print("\n✅ Document fully indexed.")

