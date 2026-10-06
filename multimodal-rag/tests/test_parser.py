# tests/test_parser.py

from ingestion import parse_pdf
def test_parse_pdf():
# use any PDF you have locally
    result = parse_pdf(r"C:\Users\RAMASUBRAMANIAN\Desktop\Travelling\Nayana_12_JAN_26.pdf")

    print("\n── TEXT CHUNKS ──")
    for chunk in result["text_chunks"][:2]:  # print first 2
        print(f"  Page {chunk['page']}: {chunk['content'][:100]}...")

    print("\n── IMAGES ──")
    for img in result["images"]:
        print(f"  Page {img['page']}: {img['path']}")

    print("\n── TABLES ──")
    for table in result["tables"]:
        print(f"  Page {table['page']}:\n{table['content'][:200]}")
    return result


if __name__ == "__main__":
    # Run the test
    test_parse_pdf()
