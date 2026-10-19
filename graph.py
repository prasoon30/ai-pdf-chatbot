from state import State
from langgraph.graph import StateGraph, START, END
from nodes import classifier_node,general_node,pdf_rag_node,response_node,web_node


## router function
def route_query(state: State) :
    if state['query_type'] == 'pdf':
        return "pdf_rag"
    elif state['query_type'] == 'web':
        return "web"
    else:
        return "general"


def route_after_pdf(state: State):
    """Route to web search if the PDF context is insufficient."""
    if state.get("needs_web", False):
        return "web"
    return "response"

# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("classifier", classifier_node)
graph.add_node("pdf_rag", pdf_rag_node)
graph.add_node("web", web_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

# Start with classification
graph.add_edge(START, "classifier")

# Route initial query
graph.add_conditional_edges(
    "classifier",
    route_query,
    {
        "pdf_rag": "pdf_rag",
        "web": "web",
        "general": "general"
    }
)

# If PDF context is insufficient, search the web
graph.add_conditional_edges(
    "pdf_rag",
    route_after_pdf,
    {
        "web": "web",
        "response": "response"
    }
)

# All paths generate a final answer
graph.add_edge("web", "response")
graph.add_edge("general", "response")
graph.add_edge("response", END)

# Compile
app = graph.compile()



