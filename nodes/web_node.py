import os
from tavily import TavilyClient
from state import State

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv()
        _client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _client


def web_node(state: State) -> dict:
    """Search the web using Tavily."""

    query = state["messages"][-1].content

    try:
        response = _get_client().search(
            query=query,
            search_depth="basic",
            max_results=5,
        )
    except Exception as e:
        return {"web_search_results": f"Web search failed: {e}"}

    results = []
    for item in response.get("results", []):
        results.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
        )

    return {"web_search_results": "\n\n".join(results)}