import os
import json
import streamlit as st

from graph import app
from utils.vector_store import sync_vectorstore
from utils.pdf_processor import (
    load_pdf_summaries,
    pdf_summaries,
    PDF_FOLDER,
    SUMMARY_FILE,
)

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="centered"
)

os.makedirs(PDF_FOLDER, exist_ok=True)


def refresh_summaries():
    """Re-run summarization for new PDFs and update the shared dict in place,
    so the classifier node (which imported the same dict) sees the changes."""
    new_summaries = load_pdf_summaries()
    pdf_summaries.clear()
    pdf_summaries.update(new_summaries)


def delete_pdf(filename: str):
    path = os.path.join(PDF_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
    pdf_summaries.pop(filename, None)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(pdf_summaries, f, indent=4, ensure_ascii=False)
    sync_vectorstore()


# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "store_synced" not in st.session_state:
    # Index any PDFs already in pdf_saver/ (first run only)
    with st.spinner("Indexing PDFs..."):
        sync_vectorstore()
    st.session_state.store_synced = True

# ---------------- Sidebar: PDF management ----------------
with st.sidebar:
    st.header("📄 Your PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files and st.button("📥 Process PDFs", use_container_width=True):
        saved = 0
        for uploaded in uploaded_files:
            save_path = os.path.join(PDF_FOLDER, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            saved += 1

        with st.spinner("Summarizing and indexing new PDFs..."):
            refresh_summaries()
            sync_vectorstore()

        # Reset the uploader so files aren't re-saved on every rerun
        st.session_state.uploader_key += 1
        st.success(f"Added {saved} PDF(s).")
        st.rerun()

    st.divider()

    existing_pdfs = sorted(
        f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")
    )

    if existing_pdfs:
        for pdf_name in existing_pdfs:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"📘 {pdf_name}")
            if col2.button("🗑️", key=f"del_{pdf_name}", help=f"Remove {pdf_name}"):
                delete_pdf(pdf_name)
                st.rerun()
    else:
        st.info("No PDFs yet. Upload some to chat with them.")

    st.divider()
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main chat area ----------------
st.title("📚 AI PDF Chatbot")
st.caption("Chat with your PDFs, search the web, or ask general questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask something...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({
                    "messages": [{"role": "user", "content": user_query}],
                    "programme": "",
                    "query_type": "",
                    "retrieved_context": "",
                    "pdf_path": "",
                    "web_search_results": "",
                    "needs_web": False,
                })

                answer = result["messages"][-1].content
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")