import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from tavily import TavilyClient

from graph.state import ResearchState
from graph.schemas import QueryDecomposition, Finding, ResearchEvaluation, ResearchReport


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

# LLM

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# STRUCTURED OUTPUT
structured_llm = llm.with_structured_output(
    QueryDecomposition,
    method="json_mode"
)

finding_llm = llm.with_structured_output(
    Finding,
    method="json_mode"
)

evaluation_llm = llm.with_structured_output(
    ResearchEvaluation,
    method="json_mode"
)


# TAVILY
tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)

# DECOMPOSE RESEARCH QUESTION

def decompose_node(state: ResearchState):

    query = state["query"]

    print("\n" + "=" * 70)
    print("DECOMPOSING RESEARCH QUESTION")
    print("=" * 70)

    print("\nOriginal question:")
    print(query)

    prompt = f"""
You are a research planning assistant.

Break the following research question into 3 to 5
focused and independent research questions.

Original research question:

{query}

Return ONLY valid JSON in this exact format:

{{
    "sub_questions": [
        "question 1",
        "question 2",
        "question 3"
    ]
}}

Rules:
- Return 3 to 5 questions.
- Each question should investigate a different aspect.
- Questions should be specific and researchable.
- Do not include explanations.
- Do not include markdown.
"""

    result = structured_llm.invoke(prompt)

    print("\nGenerated sub-questions:")

    for i, question in enumerate(result.sub_questions, start=1):
        print(f"{i}. {question}")

    return {
        "sub_questions": result.sub_questions
    }

#  RESEARCH
def research_node(state: ResearchState):

    research_round = state.get("research_round", 0) + 1

    print("\n" + "=" * 70)
    print(f"RESEARCH ROUND {research_round}")
    print("=" * 70)

    # Round 1 → use decomposed questions
    # Later rounds → use gaps discovered by evaluator
    if research_round == 1:
        questions = state["sub_questions"]
    else:
        questions = state.get("gaps", [])

    print("\nResearch questions for this round:")

    for question in questions:
        print("-", question)

    previous_findings = state.get("findings", [])

    new_findings = []

    for question in questions:

        print("\nRESEARCHING:")
        print(question)

        try:
            response = tavily_client.search(
                query=question,
                search_depth="advanced",
                max_results=5
            )

            results = response.get("results", [])

            if not results:
                print("No results found.")
                continue

            for result in results[:3]:

                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")

                prompt = f"""
You are a research evidence extractor.

Research question:
{question}

Source title:
{title}

Source URL:
{url}

Source content:
{content}

Extract the main factual claim supported by this source.

Return ONLY valid JSON in this exact format:

{{
    "claim": "main factual claim",
    "evidence": "specific evidence supporting the claim"
}}

Rules:
- Do not invent information.
- Only use information present in the source.
- Keep the claim concise.
- Keep the evidence concise.
"""

                finding = finding_llm.invoke(prompt)

                new_findings.append({
                    "question": question,
                    "claim": finding.claim,
                    "evidence": finding.evidence,
                    "source_title": title,
                    "url": url
                })

        except Exception as e:

            print("\nResearch error:")
            print(e)

    all_findings = previous_findings + new_findings

    print("\nTotal findings:", len(all_findings))

    return {
        "findings": all_findings,
        "research_round": research_round
    }

# EVALUATION NODE

def evaluate_node(state: ResearchState):

    query = state["query"]
    findings = state.get("findings", [])

    print("\n" + "=" * 70)
    print("EVALUATING RESEARCH")
    print("=" * 70)

    if not findings:

        print("\nNo findings available.")

        return {
            "sufficient": False,
            "gaps": ["No research findings were retrieved."]
        }

    findings_text = ""

    for i, finding in enumerate(findings, start=1):

        findings_text += f"""
Finding {i}

Research question:
{finding["question"]}

Claim:
{finding["claim"]}

Evidence:
{finding["evidence"]}

Source:
{finding["source_title"]}

URL:
{finding["url"]}

-------------------------
"""

    prompt = f"""
You are a research quality evaluator.

Original research question:

{query}

Current research findings:

{findings_text}

Evaluate whether the current research is sufficient to
answer the original research question.

Return ONLY valid JSON:

{{
    "sufficient": true,
    "gaps": [],
    "reasoning": "The research sufficiently covers..."
}}

OR:

{{
    "sufficient": false,
    "gaps": [
        "Missing information about..."
    ],
    "reasoning": "The research is insufficient because..."
}}

Rules:

- Mark sufficient=true only when the evidence adequately
  addresses the original research question.
- Identify important missing information.
- Do not invent facts.
- If sources do not directly answer part of the question,
  identify that as a gap.
- Keep gaps specific and researchable.
"""

    evaluation = evaluation_llm.invoke(prompt)

    print("\nResearch sufficient:")
    print(evaluation.sufficient)

    print("\nReasoning:")
    print(evaluation.reasoning)

    print("\nResearch gaps:")

    for gap in evaluation.gaps:
        print("-", gap)

    return {
        "sufficient": evaluation.sufficient,
        "gaps": evaluation.gaps,
    }



#SYNTHESIS NODE
def synthesis_node(state: ResearchState):

    print("\n" + "=" * 70)
    print("SYNTHESIZING FINAL REPORT")
    print("=" * 70)

    query = state["query"]
    findings = state.get("findings", [])

    findings_text = ""

    for i, finding in enumerate(findings, start=1):

        findings_text += f"""
Finding {i}

Research Question:
{finding.get("question", "")}

Claim:
{finding.get("claim", "")}

Evidence:
{finding.get("evidence", "")}

Source:
{finding.get("source_title", "")}

URL:
{finding.get("url", "")}

-----------------------------------
"""

    prompt = f"""
You are an expert research analyst.

Original research question:

{query}

Research findings:

{findings_text}

Write a final research report answering the original question.

Use exactly this structure:

RESEARCH SUMMARY
Give a concise answer to the original question.

KEY FINDINGS
Explain the most important findings.

COMPARISON
Compare the relevant approaches discussed in the research.

LIMITATIONS
Explain important gaps or limitations in the available evidence.

CONCLUSION
Give a balanced conclusion based only on the evidence.

SOURCES
List the URLs of the sources used.

IMPORTANT RULES:

- Use ONLY the supplied research findings.
- Do not invent facts.
- Do not invent statistics.
- Do not invent sources.
- Do not claim that evidence exists when it does not.
- Clearly state when the research does not directly answer
  part of the original question.
- Preserve the source URLs.
"""

    response = llm.invoke(prompt)

    final_report = response.content

    print("\nFINAL REPORT:\n")
    print(final_report)

    return {
        "final_report": final_report
    }

# ROUTING FUNC
MAX_RESEARCH_ROUNDS = 2


def route_after_evaluation(state: ResearchState):

    sufficient = state.get("sufficient", False)

    research_round = state.get("research_round", 0)

    if sufficient:
        return "sufficient"

    if research_round >= MAX_RESEARCH_ROUNDS:
        print("\nMaximum research rounds reached.")
        return "sufficient"

    return "needs_research"


# DISPLAY FINDINGS
def display_findings_node(state: ResearchState):

    findings = state.get("findings", [])

    print("\n" + "=" * 70)
    print("STRUCTURED FINDINGS")
    print("=" * 70)

    if not findings:

        print("\nNo findings were generated.")

        return {
            "findings": []
        }

    for i, finding in enumerate(findings, start=1):

        print("\n" + "-" * 70)
        print(f"FINDING {i}")
        print("-" * 70)

        print("\nResearch question:")
        print(finding["question"])

        print("\nClaim:")
        print(finding["claim"])

        print("\nEvidence:")
        print(finding["evidence"])

        print("\nSource:")
        print(finding["source_title"])

        print("\nURL:")
        print(finding["url"])

    return {}


# BUILD LANGGRAPH
graph_builder = StateGraph(ResearchState)

graph_builder.add_node(
    "decompose",
    decompose_node
)

graph_builder.add_node(
    "research",
    research_node
)

graph_builder.add_node(
    "evaluate",
    evaluate_node
)

graph_builder.add_node(
    "synthesis",
    synthesis_node
)

graph_builder.add_node(
    "display_findings",
    display_findings_node
)


# EDGES
graph_builder.add_edge(
    START,
    "decompose"
)

graph_builder.add_edge(
    "decompose",
    "research"
)

graph_builder.add_edge(
    "research",
    "evaluate"
)

graph_builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "sufficient": "synthesis",
        "needs_research": "research",
    }
)

graph_builder.add_edge(
    "synthesis",
    END
)

# COMPILE
graph = graph_builder.compile()

# INITIAL STATE
initial_state = {
    "query": (
        "How do user satisfaction and engagement metrics vary "
        "when customers interact with RAG-based versus fine-tuned "
        "support systems in an enterprise setting?"
    ),

    "sub_questions": [],

    "findings": [],

    "gaps":[],

    "sufficient":False,

    "research_round":0,

    "final_report": ""
}

# RUN GRAPH
if __name__ == "__main__":

    result = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)

    print(result)