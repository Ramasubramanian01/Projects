"""Ingestion pipeline for parsing and embedding raw documents."""
from .pdf_parser import parse_pdf
from .indexer import index_document
from .embedder import embed_text, embed_image, embed_query, embed_query_for_image, embed_texts_batch
__all__ = ["parse_pdf", "embed_text", "embed_image", "embed_query", "embed_query_for_image", "index_document", "embed_texts_batch"]