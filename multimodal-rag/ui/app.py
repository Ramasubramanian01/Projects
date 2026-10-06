# ui/app.py

import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Multimodal RAG")
st.caption("Upload a PDF — ask questions across text, tables and images")
st.divider()

# ── Session state ─────────────────────────────────────────────────
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "filename" not in st.session_state:
    st.session_state.filename = ""
if "history" not in st.session_state:
    st.session_state.history = []


# ── Upload Section ────────────────────────────────────────────────
st.subheader("Step 1 — Upload your PDF")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file and not st.session_state.uploaded:
    with st.spinner("Parsing and indexing your PDF..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )

    if response.status_code == 200:
        data = response.json()
        st.session_state.uploaded = True
        st.session_state.filename = uploaded_file.name
        st.success(f"✅ Indexed **{uploaded_file.name}** successfully")
        col1, col2, col3 = st.columns(3)
        col1.metric("Text Chunks", data["text_chunks"])
        col2.metric("Images",      data["images"])
        col3.metric("Tables",      data["tables"])
    else:
        st.error(f"Upload failed: {response.text}")

if st.session_state.uploaded:
    st.info(f"📎 Active document: **{st.session_state.filename}**")

st.divider()


# ── Query Section ─────────────────────────────────────────────────
st.subheader("Step 2 — Ask a question")

if not st.session_state.uploaded:
    st.warning("Upload a PDF first to start asking questions.")
else:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the key findings on page 3?"
    )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Searching document and generating answer..."):
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question}
            )

        if response.status_code == 200:
            data = response.json()

            # save to history
            st.session_state.history.append({
                "question": data["question"],
                "answer"  : data["answer"],
                "pages"   : data["cited_pages"],
                "sources" : data["sources"]
            })
        else:
            st.error(f"Query failed: {response.text}")

    # ── Chat history ──────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        st.subheader("Conversation")

        for item in reversed(st.session_state.history):
            with st.chat_message("user"):
                st.write(item["question"])

            with st.chat_message("assistant"):
                st.write(item["answer"])
                if item["pages"]:
                    st.caption(
                        f"📖 Sources — Pages: {item['pages']} | "
                        f"Text: {item['sources']['text_chunks']} chunks | "
                        f"Tables: {item['sources']['tables']} | "
                        f"Images: {item['sources']['images']}"
                    )

        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()