import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import SupportState
from graph.schemas import TicketClassification

load_dotenv()

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

structured_llm=llm.with_structured_output(TicketClassification)
CONFIDENCE_THRESHOLD = 0.95

#intake node
def intake_node(state: SupportState):
    ticket=state["ticket"]
    prompt = f"""
You are a customer support ticket classifier.

Analyze this customer ticket:

{ticket}

Classify it according to the following rules:

Category:
- technical
- billing
- general

Urgency:
- low
- medium
- high
- critical

Also identify:
- language
- short summary
- confidence between 0 and 1
"""
    result=structured_llm.invoke(prompt)
    print(result)
    return{
        "category": result.category,
        "urgency": result.urgency,
        "language": result.language,
        "summary": result.summary,
        "confidence": result.confidence,
    }


#specialist nodes
def technical_node(state: SupportState):
    print("\nTECHNICAL SPECIALIST")
    prompt = f"""
You are a technical customer support specialist.

Customer ticket:
{state["ticket"]}

Ticket summary:
{state["summary"]}

Urgency:
{state["urgency"]}

Provide a helpful technical support response.

Rules:
- Be clear and concise.
- Do not invent information.
- Give practical troubleshooting steps when appropriate.
- If you do not have enough information, ask the customer for the missing information.
"""
    response = llm.invoke(prompt)
    return {
        "response": response.content
    }

def billing_node(state: SupportState):
    print("\nBILLING SPECIALIST")
    prompt = f"""
You are a billing customer support specialist.

Customer ticket:
{state["ticket"]}

Ticket summary:
{state["summary"]}

Urgency:
{state["urgency"]}

Provide a helpful billing-related response.

Rules:
- Be professional and empathetic.
- Clearly acknowledge the customer's billing concern.
- Do not claim that a refund or payment reversal has happened.
- Do not invent transaction information.
- Explain what the customer should do next.
"""
    response = llm.invoke(prompt)
    return {
        "response": response.content
    }

def general_node(state: SupportState):
   print("\nGENERAL SUPPORT SPECIALIST")
   prompt = f"""
You are a general customer support specialist.

Customer ticket:
{state["ticket"]}

Ticket summary:
{state["summary"]}

Urgency:
{state["urgency"]}

Provide a helpful response to the customer.

Rules:
- Be friendly and professional.
- Answer the customer's question directly.
- Keep the response concise.
- If information is missing, ask an appropriate follow-up question.
"""
   response = llm.invoke(prompt)
   return {
        "response": response.content
    }

#confidence check
def check_confidence_node(state:SupportState):
    confidence=state["confidence"]
    return {}

#confidence routing
def route_by_confidence(state: SupportState):
    confidence = state["confidence"]
    if confidence >= CONFIDENCE_THRESHOLD:
        return "confident"
    return "uncertain"

#human review node
def human_review_node(state: SupportState):

    print("\nHUMAN REVIEW REQUIRED")
    decision = interrupt({
        "message": "Please review this customer support ticket.",
        "ticket": state["ticket"],
        "category": state["category"],
        "urgency": state["urgency"],
        "summary": state["summary"],
        "confidence": state["confidence"],
    })
    return {
        "response": decision
    }


def route_human_decision(state: SupportState):
    decision = state["human_decision"].lower().strip()
    if "approve" in decision:
        return "approve"
    elif "reject" in decision:
        return "reject"
    elif "edit" in decision:
        return "edit"
    else:
        return "reject"


    
#routing funcs
def route_tickets(state:SupportState):
    category=state["category"]
    if category=="technical":
        return "technical"
    elif category=="billing":
        return "billing"
    else:
        return "general"



    
graph_builder = StateGraph(SupportState)

graph_builder.add_node("intake", intake_node)
graph_builder.add_node("technical", technical_node)
graph_builder.add_node("billing", billing_node)
graph_builder.add_node("general", general_node)
graph_builder.add_node("confidence_check", check_confidence_node)
graph_builder.add_node("human_review", human_review_node)

graph_builder.add_edge(START,"intake")
graph_builder.add_conditional_edges(
    "intake",
    route_tickets,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general",
    }
)

graph_builder.add_edge("technical","confidence_check")
graph_builder.add_edge("billing", "confidence_check")
graph_builder.add_edge("general", "confidence_check")

graph_builder.add_conditional_edges(
    "confidence_check",
    route_by_confidence,
    {
        "confident": END,
        "uncertain": "human_review",
    }
)
graph_builder.add_edge(
    "human_review",
    END
)
checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)

initial_state = {
    "ticket": "The application crashes whenever I try to upload a PDF.",
    "category": "",
    "urgency": "",
    "language": "",
    "summary": "",
    "confidence": 0.0,
    "response": "",
    "human_decision":"",
}

#thred config
config = {
    "configurable": {
        "thread_id": "ticket-001"
    }
}

#first graph invocation
result = graph.invoke(
    initial_state,
    config=config
)

print("\nGRAPH PAUSED / FINISHED")
print(result)
current_state = graph.get_state(config)
print("\nCURRENT STATE")
print(current_state)

#human resume
if current_state.next:
    print("\nThe graph is waiting for human review.")
    human_decision = input(
        "\nEnter human decision: "
    )
    result = graph.invoke(
        Command(
            resume=human_decision
        ),
        config=config
    )
    print("\n--- FINAL STATE ---")
    print(result)

else:
    print("\nGraph completed without human review.")