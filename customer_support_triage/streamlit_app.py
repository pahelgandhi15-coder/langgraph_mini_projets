import uuid

import streamlit as st

from langgraph.types import Command

from app import graph


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Support Triage",
    page_icon="🎧",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "result" not in st.session_state:
    st.session_state.result = None

if "waiting_for_human" not in st.session_state:
    st.session_state.waiting_for_human = False

if "ticket" not in st.session_state:
    st.session_state.ticket = ""


# ==========================================
# TITLE
# ==========================================

st.title("🎧 Customer Support Triage System")

st.write(
    "A LangGraph-based customer support workflow "
    "with classification, specialist routing, "
    "confidence checking, and human-in-the-loop review."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Workflow")

    st.markdown(
        """
        **1. Intake**

        Classify the customer ticket.

        **2. Specialist**

        Route to technical, billing, or general support.

        **3. Confidence Check**

        Determine whether human review is required.

        **4. Human Review**

        Approve, reject, or edit the response.
        """
    )

    st.divider()

    st.write("Thread ID")

    st.code(
        st.session_state.thread_id
    )

    if st.button("New Ticket"):

        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.result = None
        st.session_state.waiting_for_human = False
        st.session_state.ticket = ""

        st.rerun()


# ==========================================
# TICKET INPUT
# ==========================================

st.subheader("Customer Ticket")

ticket = st.text_area(
    "Enter the customer's issue:",
    value=st.session_state.ticket,
    height=150,
    placeholder=(
        "Example: My payment was deducted "
        "but my subscription is still inactive."
    )
)


# ==========================================
# PROCESS TICKET
# ==========================================

if st.button(
    "🚀 Analyze Ticket",
    type="primary"
):

    if not ticket.strip():

        st.warning(
            "Please enter a customer support ticket."
        )

    else:

        st.session_state.ticket = ticket

        initial_state = {
            "ticket": ticket,
            "category": "",
            "urgency": "",
            "language": "",
            "summary": "",
            "confidence": 0.0,
            "response": "",
            "human_decision": "",
        }

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        try:

            with st.spinner(
                "Analyzing customer ticket..."
            ):

                result = graph.invoke(
                    initial_state,
                    config=config
                )

            st.session_state.result = result

            current_state = graph.get_state(
                config
            )

            if current_state.next:

                st.session_state.waiting_for_human = True

            else:

                st.session_state.waiting_for_human = False

            st.rerun()

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# ==========================================
# DISPLAY RESULT
# ==========================================

if st.session_state.result:

    result = st.session_state.result

    st.divider()

    st.subheader("📊 Ticket Analysis")


    # ======================================
    # METRICS
    # ======================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Category",
            result["category"].upper()
        )

    with col2:

        st.metric(
            "Urgency",
            result["urgency"].upper()
        )

    with col3:

        st.metric(
            "Language",
            result["language"]
        )

    with col4:

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}"
        )


    # ======================================
    # SUMMARY
    # ======================================

    st.subheader("📝 Summary")

    st.info(
        result["summary"]
    )


    # ======================================
    # RESPONSE
    # ======================================

    st.subheader("💬 Suggested Response")

    st.write(
        result["response"]
    )


# ==========================================
# HUMAN REVIEW
# ==========================================

if st.session_state.waiting_for_human:

    st.divider()

    st.subheader("👨‍💼 Human Review Required")

    result = st.session_state.result

    st.warning(
        "The model confidence is below the "
        "required threshold. Please review "
        "the generated response."
    )


    # ======================================
    # REVIEW INFORMATION
    # ======================================

    st.write("### Ticket")

    st.write(
        result["ticket"]
    )

    st.write("### Classification")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Category:** {result['category']}"
        )

        st.write(
            f"**Urgency:** {result['urgency']}"
        )

    with col2:

        st.write(
            f"**Confidence:** "
            f"{result['confidence']:.2f}"
        )

        st.write(
            f"**Language:** {result['language']}"
        )


    # ======================================
    # DECISION
    # ======================================

    st.write("### Review Decision")

    decision = st.radio(
        "Choose an action:",
        [
            "approve",
            "reject",
            "edit"
        ],
        horizontal=True
    )


    # ======================================
    # EDIT RESPONSE
    # ======================================

    if decision == "edit":

        edited_response = st.text_area(
            "Edit the response:",
            value=result["response"],
            height=200
        )

    else:

        edited_response = result["response"]


    # ======================================
    # SUBMIT DECISION
    # ======================================

    if st.button(
        "Submit Review",
        type="primary"
    ):

        config = {
            "configurable": {
                "thread_id":
                    st.session_state.thread_id
            }
        }

        try:

            with st.spinner(
                "Updating workflow..."
            ):

                if decision == "edit":

                    # For now the graph's edit node
                    # regenerates the response.
                    resume_value = "edit"

                else:

                    resume_value = decision


                result = graph.invoke(
                    Command(
                        resume=resume_value
                    ),
                    config=config
                )


            st.session_state.result = result

            st.session_state.waiting_for_human = False

            st.rerun()

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# ==========================================
# FINAL STATUS
# ==========================================

if (
    st.session_state.result
    and not st.session_state.waiting_for_human
):

    st.divider()

    st.success(
        "✅ Support workflow completed."
    )

    if st.session_state.result.get(
        "human_decision"
    ):

        st.write(
            "**Human Decision:** "
            + st.session_state.result[
                "human_decision"
            ]
        )