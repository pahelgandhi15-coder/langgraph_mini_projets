from pydantic import BaseModel,Field

class TicketClassification(BaseModel):
    category:str=Field(
        description="Ticket category : technical,billing, or general"
    )
    
    urgency: str = Field(
        description="Ticket urgency: low, medium, high, or critical"
    )

    language: str = Field(
        description="Language used by the customer"
    )

    summary: str = Field(
        description="Short one-sentence summary of the customer issue"
    )

    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )