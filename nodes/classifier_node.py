from state import State
from llm import llm
import json

from utils.pdf_processor import pdf_summaries

def classifier_node(state: State) -> dict:
    """Classify the query using uploaded PDF summaries."""

    last_message = state["messages"][-1].content

    # Get summaries of all uploaded PDFs
    summaries_text = json.dumps(
        pdf_summaries,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are a query classifier for a chatbot with three capabilities:
1. Answer questions using uploaded PDFs.
2. Search the web using Tavily.
3. Answer general questions using an LLM.

Here are the summaries of the uploaded PDFs:
{summaries_text}

Classify the user's query into exactly ONE category:

PDF:
- Choose "pdf" if the question can be answered using information
  from any uploaded PDF.
- Match the question against the PDF filenames and summaries.
- Includes questions about topics, definitions, rules, facts,
  chapters, examples, or details contained in uploaded documents.
- Follow-up questions about previously discussed PDF content
  should also use "pdf".
- If multiple PDFs are relevant, choose "pdf". Retrieval can
  search across all uploaded PDFs.

WEB:
- Choose "web" if the user explicitly requests online research,
  current information, recent news, or information that needs
  external web verification.
- Examples: latest AI news, recent research papers, current prices.

GENERAL:
- Choose "general" for greetings, casual conversation,
  coding help, and self-contained questions unrelated to
  uploaded PDFs or current external information.

Rules:
- Prefer "pdf" when the query matches an uploaded PDF summary.
- Prefer "web" when the user explicitly asks to search online.
- Return only one category word.

User query:
{last_message}

Category:
"""

    response = llm.invoke(prompt)
    query_type = response.content.strip().lower()

    # Validate the classifier output
    if query_type not in ["pdf", "web", "general"]:
        query_type = "general"

    return {"query_type": query_type}
