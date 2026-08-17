from typing import TypedDict


class ResearchState(TypedDict):

    query: str

    sub_questions: list[str]

    findings: list[dict]

    gaps: list[str]

    sufficient: bool

    research_round: int

    final_report: str