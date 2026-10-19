import os
import json
from langchain_community.document_loaders import PyPDFLoader
from llm import llm

PDF_FOLDER = "pdf_saver"
SUMMARY_FILE = "pdf_summaries.json"


def load_pdf_summaries():
    summaries = {}

    # Load existing summaries if available
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summaries = json.load(f)

    # Generate summaries for PDFs not yet summarized
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

    for filename in os.listdir(PDF_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        if filename in summaries:
            continue

        pdf_path = os.path.join(PDF_FOLDER, filename)
        documents = PyPDFLoader(pdf_path).load()

        text = "\n".join(doc.page_content for doc in documents)
        text = text[:25000]

        if not text.strip():
            summaries[filename] = "No readable text found."
            continue

        prompt = f"""
Summarize this PDF for a chatbot query classifier.
Identify its main topics, important facts, definitions,
sections, and the types of questions it can answer.
Be concise but informative.

PDF filename: {filename}

PDF content:
{text}
"""

        response = llm.invoke(prompt)
        summaries[filename] = response.content

        # Save after each PDF
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4, ensure_ascii=False)

    # Also create the JSON file when no PDFs exist
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4, ensure_ascii=False)

    return summaries


pdf_summaries = load_pdf_summaries()