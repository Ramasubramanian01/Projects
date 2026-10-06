# ingestion/pdf_parser.py

import fitz  # PyMuPDF
import os
from pathlib import Path


def extract_text_chunks(pdf_path: str, chunk_size: int = 500) -> list[dict]:
    """
    Extract text from PDF and split into chunks.
    Each chunk carries metadata: page number, chunk index.
    """
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        if not text:
            continue

        # split into chunks of ~chunk_size characters
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size:
                chunks.append({
                    "type": "text",
                    "content": " ".join(current_chunk),
                    "page": page_num + 1,
                    "chunk_index": len(chunks)
                })
                current_chunk = []
                current_length = 0

        # catch remaining words
        if current_chunk:
            chunks.append({
                "type": "text",
                "content": " ".join(current_chunk),
                "page": page_num + 1,
                "chunk_index": len(chunks)
            })

    doc.close()
    return chunks


def extract_images(pdf_path: str, output_dir: str = "extracted_images") -> list[dict]:
    """
    Extract images from PDF and save them to disk.
    Returns metadata list with image path and page number.
    """
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    images = []

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            images.append({
                "type": "image",
                "path": image_path,
                "page": page_num + 1,
                "image_index": img_index + 1
            })

    doc.close()
    return images


def extract_tables(pdf_path: str) -> list[dict]:
    """
    Extract tables from PDF pages.
    Returns table content as plain text with page reference.
    """
    doc = fitz.open(pdf_path)
    tables = []

    for page_num, page in enumerate(doc):
        # PyMuPDF table extraction
        table_finder = page.find_tables()

        for table_index, table in enumerate(table_finder.tables):
            rows = table.extract()

            # convert rows to readable string
            table_text = ""
            for row in rows:
                cleaned = [str(cell).strip() if cell else "" for cell in row]
                table_text += " | ".join(cleaned) + "\n"

            if table_text.strip():
                tables.append({
                    "type": "table",
                    "content": table_text.strip(),
                    "page": page_num + 1,
                    "table_index": table_index + 1
                })

    doc.close()
    return tables


def parse_pdf(pdf_path: str, image_output_dir: str = "extracted_images") -> dict:
    """
    Master function — runs all three extractors and returns
    a single dictionary with text chunks, images, and tables.
    """
    print(f"Parsing PDF: {pdf_path}")

    text_chunks = extract_text_chunks(pdf_path)
    images      = extract_images(pdf_path, output_dir=image_output_dir)
    tables      = extract_tables(pdf_path)

    print(f"  Text chunks : {len(text_chunks)}")
    print(f"  Images      : {len(images)}")
    print(f"  Tables      : {len(tables)}")

    return {
        "text_chunks": text_chunks,
        "images": images,
        "tables": tables
    }