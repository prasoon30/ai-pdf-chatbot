from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    programme: str
    query_type: str
    retrieved_context: str
    pdf_path: str
    web_search_results: str
    needs_web: bool