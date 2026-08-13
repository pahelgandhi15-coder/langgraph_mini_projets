from typing_extensions import TypedDict


class SupportState(TypedDict):
    ticket: str
    category: str
    urgency: str
    language: str
    summary: str
    confidence: float
    response: str