import os
from state import State
from llm import llm
from utils.vector_store import build_retriever


def pdf_rag_node(state: State) -> dict:
    """Retrieve PDF context and check whether web fallback is needed."""

    query = state["messages"][-1].content

    retriever = build_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return {
            "retrieved_context": "",
            "web_search_results": "",
            "needs_web": True
        }

    context_parts = []

    for doc in documents:
        source = os.path.basename(
            doc.metadata.get("source", "Unknown document")
        )
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    check_prompt = f"""
You are checking whether retrieved PDF information is sufficient
to answer a user's question.

Question:
{query}

Retrieved PDF context:
{context}

Return YES if the context contains enough relevant information
to answer the question accurately.

Return NO if the context is missing important details,
does not answer the question, or is insufficient.

Return only YES or NO.
"""

    result = llm.invoke(check_prompt).content.strip().upper()
    needs_web = not result.startswith("YES")

    return {
        "retrieved_context": context,
        "web_search_results": "",
        "needs_web": needs_web
    }