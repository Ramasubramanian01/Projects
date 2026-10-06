# retrieval/retriever.py

import chromadb
from ingestion.embedder import embed_query, embed_query_for_image

# ── ChromaDB client ──────────────────────────────────────────────
CLIENT = chromadb.PersistentClient(path="./chroma_store")

TEXT_COLLECTION  = CLIENT.get_or_create_collection(
    name="text_chunks",
    metadata={"hnsw:space": "cosine"}
)
IMAGE_COLLECTION = CLIENT.get_or_create_collection(
    name="image_chunks",
    metadata={"hnsw:space": "cosine"}
)
TABLE_COLLECTION = CLIENT.get_or_create_collection(
    name="table_chunks",
    metadata={"hnsw:space": "cosine"}
)


def retrieve_text(query: str, top_k: int = 3) -> list[dict]:
    """
    Search text collection for relevant chunks.
    Returns top_k results with content, page, and score.
    """
    if TEXT_COLLECTION.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = TEXT_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, TEXT_COLLECTION.count())
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "type"    : "text",
            "content" : doc,
            "page"    : results["metadatas"][0][i]["page"],
            "score"   : round(1 - results["distances"][0][i], 3)
        })

    return chunks


def retrieve_images(query: str, top_k: int = 2) -> list[dict]:
    """
    Search image collection using CLIP text encoder.
    Returns top_k image paths with page reference.
    """
    if IMAGE_COLLECTION.count() == 0:
        return []

    query_embedding = embed_query_for_image(query)

    results = IMAGE_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, IMAGE_COLLECTION.count())
    )

    images = []
    for i, doc in enumerate(results["documents"][0]):
        images.append({
            "type"  : "image",
            "path"  : results["metadatas"][0][i]["path"],
            "page"  : results["metadatas"][0][i]["page"],
            "score" : round(1 - results["distances"][0][i], 3)
        })

    return images


def retrieve_tables(query: str, top_k: int = 2) -> list[dict]:
    """
    Search table collection for relevant tables.
    Returns top_k tables with content and page reference.
    """
    if TABLE_COLLECTION.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = TABLE_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, TABLE_COLLECTION.count())
    )

    tables = []
    for i, doc in enumerate(results["documents"][0]):
        tables.append({
            "type"    : "table",
            "content" : doc,
            "page"    : results["metadatas"][0][i]["page"],
            "score"   : round(1 - results["distances"][0][i], 3)
        })

    return tables


def retrieve_all(query: str) -> dict:
    """
    Master retriever — searches all three collections.
    Returns combined results with citation metadata.
    """
    print(f"\nRetrieving context for: '{query}'")

    text_results  = retrieve_text(query,   top_k=3)
    image_results = retrieve_images(query, top_k=2)
    table_results = retrieve_tables(query, top_k=2)

    print(f"  Text  : {len(text_results)} chunks")
    print(f"  Images: {len(image_results)} images")
    print(f"  Tables: {len(table_results)} tables")

    return {
        "query"  : query,
        "text"   : text_results,
        "images" : image_results,
        "tables" : table_results
    }