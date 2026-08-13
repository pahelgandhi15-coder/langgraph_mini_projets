from langgraph.graph import StateGraph, START, END

from graph.state import SupportState


def intake_node(state: SupportState):
    print("\n--- INTAKE NODE ---")

    print("Customer ticket:")
    print(state["ticket"])

    return {
        "category": "billing",
        "urgency": "medium",
        "language": "English",
        "summary": "Customer is having a billing-related issue.",
        "confidence": 0.90,
    }

graph_builder = StateGraph(SupportState)

graph_builder.add_node("intake", intake_node)

graph_builder.add_edge(START, "intake")
graph_builder.add_edge("intake", END)

graph = graph_builder.compile()

initial_state = {
    "ticket": "My payment was deducted but my subscription is still inactive.",
    "category": "",
    "urgency": "",
    "language": "",
    "summary": "",
    "confidence": 0.0,
    "response": "",
}

result = graph.invoke(initial_state)

print("\n--- FINAL STATE ---")
print(result)