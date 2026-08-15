import os
import uuid

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from langchain_groq import ChatGroq

from graph.state import SupportState
from graph.schemas import TicketClassification


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# LLM SETUP
# ==========================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

structured_llm = llm.with_structured_output(
    TicketClassification
)


# ==========================================
# CONFIDENCE THRESHOLD
# ==========================================

CONFIDENCE_THRESHOLD = 0.75


# ==========================================
# INTAKE NODE
# ==========================================

def intake_node(state: SupportState):

    ticket = state["ticket"]

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

Return only the structured classification.
"""

    result = structured_llm.invoke(prompt)

    print("\n--- INTAKE CLASSIFICATION ---")
    print(result)

    return {
        "category": result.category,
        "urgency": result.urgency,
        "language": result.language,
        "summary": result.summary,
        "confidence": result.confidence,
    }


# ==========================================
# TECHNICAL SPECIALIST
# ==========================================

def technical_node(state: SupportState):

    print("\n--- TECHNICAL SPECIALIST ---")

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


# ==========================================
# BILLING SPECIALIST
# ==========================================

def billing_node(state: SupportState):

    print("\n--- BILLING SPECIALIST ---")

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


# ==========================================
# GENERAL SPECIALIST
# ==========================================

def general_node(state: SupportState):

    print("\n--- GENERAL SUPPORT SPECIALIST ---")

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


# ==========================================
# CATEGORY ROUTING
# ==========================================

def route_tickets(state: SupportState):

    category = state["category"]

    if category == "technical":
        return "technical"

    elif category == "billing":
        return "billing"

    else:
        return "general"


# ==========================================
# CONFIDENCE CHECK NODE
# ==========================================

def confidence_check_node(state: SupportState):

    confidence = state["confidence"]

    print("\n--- CONFIDENCE CHECK ---")
    print(f"Confidence: {confidence}")
    print(f"Threshold: {CONFIDENCE_THRESHOLD}")

    return {}


# ==========================================
# CONFIDENCE ROUTING
# ==========================================

def route_by_confidence(state: SupportState):

    confidence = state["confidence"]

    if confidence >= CONFIDENCE_THRESHOLD:
        return "confident"

    return "uncertain"


# ==========================================
# HUMAN REVIEW NODE
# ==========================================

def human_review_node(state: SupportState):

    print("\n--- HUMAN REVIEW REQUIRED ---")

    decision = interrupt({
        "message": "Please review this customer support ticket.",
        "ticket": state["ticket"],
        "category": state["category"],
        "urgency": state["urgency"],
        "summary": state["summary"],
        "confidence": state["confidence"],
        "suggested_response": state["response"],
    })

    return {
        "human_decision": decision
    }


# ==========================================
# HUMAN DECISION ROUTING
# ==========================================

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


# ==========================================
# APPROVE NODE
# ==========================================

def approve_node(state: SupportState):

    print("\n--- RESPONSE APPROVED ---")

    return {
        "response": state["response"]
    }


# ==========================================
# REJECT NODE
# ==========================================

def reject_node(state: SupportState):

    print("\n--- RESPONSE REJECTED ---")

    return {
        "response": (
            "Your request requires further investigation "
            "by our support team."
        )
    }


# ==========================================
# EDIT / REGENERATE NODE
# ==========================================

def edit_node(state: SupportState):

    print("\n--- REGENERATING RESPONSE ---")

    prompt = f"""
You are a customer support quality specialist.

Customer ticket:
{state["ticket"]}

Current support response:
{state["response"]}

The human reviewer requested an improved response.

Rewrite the response so that it is:

- Clear
- Helpful
- Professional
- Concise
- Accurate
- Based only on the information available

Do not invent information.
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content
    }


# ==========================================
# BUILD GRAPH
# ==========================================

graph_builder = StateGraph(SupportState)


# ==========================================
# ADD NODES
# ==========================================

graph_builder.add_node(
    "intake",
    intake_node
)

graph_builder.add_node(
    "technical",
    technical_node
)

graph_builder.add_node(
    "billing",
    billing_node
)

graph_builder.add_node(
    "general",
    general_node
)

graph_builder.add_node(
    "confidence_check",
    confidence_check_node
)

graph_builder.add_node(
    "human_review",
    human_review_node
)

graph_builder.add_node(
    "approve",
    approve_node
)

graph_builder.add_node(
    "reject",
    reject_node
)

graph_builder.add_node(
    "edit",
    edit_node
)


# ==========================================
# START → INTAKE
# ==========================================

graph_builder.add_edge(
    START,
    "intake"
)


# ==========================================
# INTAKE → SPECIALIST
# ==========================================

graph_builder.add_conditional_edges(
    "intake",
    route_tickets,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general",
    }
)


# ==========================================
# SPECIALIST → CONFIDENCE CHECK
# ==========================================

graph_builder.add_edge(
    "technical",
    "confidence_check"
)

graph_builder.add_edge(
    "billing",
    "confidence_check"
)

graph_builder.add_edge(
    "general",
    "confidence_check"
)


# ==========================================
# CONFIDENCE → END / HUMAN REVIEW
# ==========================================

graph_builder.add_conditional_edges(
    "confidence_check",
    route_by_confidence,
    {
        "confident": END,
        "uncertain": "human_review",
    }
)


# ==========================================
# HUMAN REVIEW → DECISION
# ==========================================

graph_builder.add_conditional_edges(
    "human_review",
    route_human_decision,
    {
        "approve": "approve",
        "reject": "reject",
        "edit": "edit",
    }
)


# ==========================================
# DECISION → END
# ==========================================

graph_builder.add_edge(
    "approve",
    END
)

graph_builder.add_edge(
    "reject",
    END
)

graph_builder.add_edge(
    "edit",
    END
)


# ==========================================
# CHECKPOINTING
# ==========================================

checkpointer = InMemorySaver()


# ==========================================
# COMPILE GRAPH
# ==========================================

graph = graph_builder.compile(
    checkpointer=checkpointer
)

# def get_ticket():

#     print("   CUSTOMER SUPPORT TRIAGE SYSTEM")
#     ticket = input(
#         "Enter customer support ticket:\n> "
#     ).strip()

#     while not ticket:
#         print("\nTicket cannot be empty.")
#         ticket = input(
#             "Enter customer support ticket:\n> "
#         ).strip()

#     return ticket

# MAIN

# ticket = get_ticket()

# # INITIAL STATE
# initial_state = {
#     "ticket": ticket,
#     "category": "",
#     "urgency": "",
#     "language": "",
#     "summary": "",
#     "confidence": 0.0,
#     "response": "",
#     "human_decision": "",
# }

# # UNIQUE THREAD ID

# config = {
#     "configurable": {
#         "thread_id": str(uuid.uuid4())
#     }
# }

# # FIRST GRAPH INVOCATION

# try:
#     result = graph.invoke(
#         initial_state,
#         config=config
#     )

# except Exception as e:

#     print("              ERROR")
#     print(e)
#     raise

# # CHECK GRAPH STATE
# current_state = graph.get_state(config)


# # HUMAN REVIEW
# if current_state.next:
#     print("        HUMAN REVIEW REQUIRED")

#     print("\nTicket:")
#     print(current_state.values["ticket"])

#     print("\nCategory:")
#     print(current_state.values["category"])

#     print("\nUrgency:")
#     print(current_state.values["urgency"])

#     print("\nSummary:")
#     print(current_state.values["summary"])

#     print("\nConfidence:")
#     print(current_state.values["confidence"])

#     print("\nSuggested Response:")
#     print(current_state.values["response"])

#     # GET HUMAN DECISION
#     human_decision = input(
#         "\nEnter decision (approve/reject/edit): "
#     ).strip().lower()


#     while human_decision not in [
#         "approve",
#         "reject",
#         "edit"
#     ]:

#         print("\nInvalid decision.")

#         human_decision = input(
#             "Enter approve, reject, or edit: "
#         ).strip().lower()

#     # RESUME GRAPH
#     try:
#         result = graph.invoke(
#             Command(
#                 resume=human_decision
#             ),
#             config=config
#         )
#     except Exception as e:
#         print("         RESUME ERROR")
#         print(e)
#         raise

# # FINAL RESULT
# print("          FINAL RESULT")

# print(f"\nCategory   : {result['category']}")
# print(f"Urgency    : {result['urgency']}")
# print(f"Language   : {result['language']}")
# print(f"Confidence : {result['confidence']}")

# print("\nSummary:")
# print(result["summary"])

# print("\nResponse:")
# print(result["response"])

# if result.get("human_decision"):

#     print("\nHuman Decision:")
#     print(result["human_decision"])

# print("          GRAPH COMPLETE")
