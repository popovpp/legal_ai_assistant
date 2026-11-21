# app/graph/builder.py
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph.nodes.analyst import make_analyst_node
from app.graph.nodes.finalize import finalize_node
from app.graph.nodes.retrieve import make_retrieve_node
from app.graph.nodes.router import route_node
from app.graph.nodes.verifier import make_verifier_node
from app.graph.state import AnalysisState
from app.llm.client import LLMClient
from app.rag.retriever import Retriever


def _route_after_verifier(state: AnalysisState) -> str:
    return "analyst" if state.get("should_retry") else "finalize"


def build_graph(retriever: Retriever, llm: LLMClient):
    g = StateGraph(AnalysisState)
    g.add_node("router", route_node)
    g.add_node("retrieve", make_retrieve_node(retriever, settings.top_k))
    g.add_node("analyst", make_analyst_node(llm))
    g.add_node("verifier", make_verifier_node())
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "router")
    g.add_edge("router", "retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_edge("analyst", "verifier")
    g.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"analyst": "analyst", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()
