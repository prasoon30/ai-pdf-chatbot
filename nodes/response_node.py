from state import State
from llm import llm

def response_node(state: State) -> dict:
    """Answer using PDF context, web results, or general knowledge."""

    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state.get("retrieved_context", "")
    web_results = state.get("web_search_results", "")

    if context.strip() or web_results.strip():
        prompt = f"""
You are a helpful AI assistant for a student in {programme}.

User question:
{query}

PDF context:
{context if context.strip() else "No relevant PDF context available."}

Web search results:
{web_results if web_results.strip() else "No web results available."}

Instructions:
- Use relevant PDF information first when available.
- Use web results to fill gaps in the PDF information.
- Clearly distinguish information from the PDFs and the web.
- Do not invent facts or citations.
- Include relevant source URLs from web results.
- If neither source provides enough evidence, say so.
- Give a clear, accurate, well-structured answer.
"""
    else:
        prompt = f"""
You are a friendly AI assistant helping a student in {programme}.

Answer this question using your general knowledge:
{query}

Be clear, accurate, and conversational. Be transparent
if the question requires information you cannot verify.
"""

    response = llm.invoke(prompt)

    return {
        "messages": [("ai", response.content.strip())]
    }
