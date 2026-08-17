from pydantic import BaseModel, Field


class QueryDecomposition(BaseModel):

    sub_questions: list[str] = Field(
        description="A list of 3 to 5 focused research questions."
    )


class Finding(BaseModel):

    claim: str = Field(
        description="The main factual claim supported by the source."
    )

    evidence: str = Field(
        description="Evidence from the source supporting the claim."
    )


class ResearchEvaluation(BaseModel):

    sufficient: bool = Field(
        description="Whether the current research is sufficient to answer the original question."
    )

    gaps: list[str] = Field(
        description="Important unanswered research questions or gaps."
    )

    reasoning: str = Field(
        default="",
        description="Brief explanation of why the research is or is not sufficient."
    )

class ResearchReport(BaseModel):

    summary: str = Field(
        description="A concise summary answering the original research question."
    )

    key_findings: list[str] = Field(
        description="The most important findings from the research."
    )

    comparison: str = Field(
        description="A comparison of the approaches discussed in the research."
    )

    limitations: list[str] = Field(
        description="Important limitations or gaps in the available evidence."
    )

    conclusion: str = Field(
        description="A balanced final conclusion based only on the evidence."
    )

    sources: list[str] = Field(
        description="URLs of the sources used in the research."
    )