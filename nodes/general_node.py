from state import State

def general_node(state: State) -> dict:
    """Marks the query as general, without retrieval."""
    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED",
        "web_search_results": ""
    }
