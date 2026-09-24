from langgraph.graph import END, START, StateGraph

from src.orchestration.nodes import (
    audit_quality,
    decide,
    extract_claims,
    generate_mcq,
    verify_answer_key,
    verify_claims,
)
from src.orchestration.state import MCQVerificationState


def route_after_decision(state: MCQVerificationState) -> str:
    decision = state["decision"]

    if decision == "REVISE":
        return "generate_mcq"

    if decision in {"ACCEPT", "REJECT"}:
        return END

    raise ValueError(f"Unknown decision: {decision}")


def build_graph():
    graph = StateGraph(MCQVerificationState)

    graph.add_node("generate_mcq", generate_mcq)
    graph.add_node("extract_claims", extract_claims)
    graph.add_node("verify_claims", verify_claims)
    graph.add_node("verify_answer_key", verify_answer_key)
    graph.add_node("audit_quality", audit_quality)
    graph.add_node("decide", decide)

    graph.add_edge(START, "generate_mcq")
    graph.add_edge("generate_mcq", "extract_claims")
    graph.add_edge("extract_claims", "verify_claims")
    graph.add_edge("verify_claims", "verify_answer_key")
    graph.add_edge("verify_answer_key", "audit_quality")
    graph.add_edge("audit_quality", "decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "generate_mcq": "generate_mcq",
            END: END,
        },
    )

    return graph.compile()
if __name__ == "__main__":
    workflow = build_graph()
    print("LangGraph workflow compiled successfully.")