import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_FOLDER = "pdf_saver"
CHROMA_DIR = "./chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Created once and reused, instead of rebuilding on every query
vectorstore = Chroma(
    collection_name="pdfs",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _indexed_sources() -> set:
    data = vectorstore.get(include=["metadatas"])
    return {m.get("source") for m in data["metadatas"] if m}


def sync_vectorstore() -> None:
    """Make the vector store match the PDFs in pdf_saver/.

    New PDFs are chunked and embedded once; deleted PDFs are removed.
    Safe to call repeatedly (no duplicate chunks).
    """
    os.makedirs(PDF_FOLDER, exist_ok=True)

    on_disk = {
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    }
    indexed = _indexed_sources()

    for path in on_disk - indexed:
        docs = PyPDFLoader(path).load()
        chunks = splitter.split_documents(docs)
        if chunks:
            vectorstore.add_documents(chunks)

    for source in indexed - on_disk:
        vectorstore.delete(where={"source": source})


def build_retriever(k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": k})