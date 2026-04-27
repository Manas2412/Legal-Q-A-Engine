from __future__ import annotations
from langgraph.graph import StateGraph, END
from graph.state import LegalQAState
from graph.nodes import (
    classify_domain_node,
    decompose_query_node,
    hyde_generator_node,
    hybrid_retriever_node,
    authority_scorer_node,
    reranker_node,
    memory_loader_node,
    generator_node,
    hallucination_guard_node,
)


def build_legal_qa_graph() -> StateGraph:
    """
    Build and compile the LangGraph state machine for the Legal Q&A pipeline.

    Flow:
    classify → decompose → hyde → [memory_load || retrieval_chain] → generate → guard
                                           ↓                  ↓
                                    memory_loader     hybrid_retrieve → authority_score → rerank
    """
    graph = StateGraph(LegalQAState)

    # Register all nodes
    graph.add_node("classify_domain",    classify_domain_node)
    graph.add_node("decompose_query",    decompose_query_node)
    graph.add_node("hyde_generator",     hyde_generator_node)
    graph.add_node("memory_loader",      memory_loader_node)
    graph.add_node("hybrid_retriever",   hybrid_retriever_node)
    graph.add_node("authority_scorer",   authority_scorer_node)
    graph.add_node("reranker",           reranker_node)
    graph.add_node("generator",          generator_node)
    graph.add_node("hallucination_guard", hallucination_guard_node)

    # Entry point
    graph.set_entry_point("classify_domain")

    # Sequential edges
    graph.add_edge("classify_domain",  "decompose_query")
    graph.add_edge("decompose_query",  "hyde_generator")
    graph.add_edge("hyde_generator",   "memory_loader")
    graph.add_edge("memory_loader",    "hybrid_retriever")
    graph.add_edge("hybrid_retriever", "authority_scorer")
    graph.add_edge("authority_scorer", "reranker")
    graph.add_edge("reranker",         "generator")
    graph.add_edge("generator",        "hallucination_guard")
    graph.add_edge("hallucination_guard", END)

    return graph.compile()


# Singleton compiled graph
legal_qa_graph = build_legal_qa_graph()


def run_query(session_id: str, query: str) -> dict:
    """
    Run a legal query through the full pipeline.
    Returns the complete final state dict.
    """
    initial_state: LegalQAState = {
        "session_id": session_id,
        "query": query,
    }

    final_state = legal_qa_graph.invoke(initial_state)
    return final_state