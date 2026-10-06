# tests/test_indexer.py

from ingestion.pdf_parser import parse_pdf
from ingestion.indexer import index_document
import chromadb

# Step 1 — parse
result = parse_pdf(r"C:\Users\RAMASUBRAMANIAN\Desktop\Travelling\Nayana_12_JAN_26.pdf")

# Step 2 — index everything
index_document(result)

# Step 3 — verify it's in ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")

text_col  = client.get_collection("text_chunks")
image_col = client.get_collection("image_chunks")
table_col = client.get_collection("table_chunks")

print(f"\n── ChromaDB Contents ──")
print(f"  Text chunks  : {text_col.count()}")
print(f"  Image chunks : {image_col.count()}")
print(f"  Table chunks : {table_col.count()}")