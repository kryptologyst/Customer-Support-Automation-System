"""
Core data structures and schemas for customer support automation.

This module defines the canonical data structures used throughout the system,
including support tickets, knowledge base entries, and customer interactions.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import pandas as pd


class TicketPriority(str, Enum):
    """Priority levels for support tickets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(str, Enum):
    """Status of support tickets."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


class IntentType(str, Enum):
    """Types of customer intents."""
    ORDER_STATUS = "order_status"
    RETURN_POLICY = "return_policy"
    CANCEL_ORDER = "cancel_order"
    PAYMENT_FAILED = "payment_failed"
    DELIVERY_TIME = "delivery_time"
    PRODUCT_INFO = "product_info"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_QUESTION = "billing_question"
    ACCOUNT_ISSUE = "account_issue"
    GENERAL_INQUIRY = "general_inquiry"


class ResponseType(str, Enum):
    """Types of automated responses."""
    AUTOMATED = "automated"
    ESCALATED = "escalated"
    HUMAN_REQUIRED = "human_required"


@dataclass
class SupportTicket:
    """Core support ticket data structure."""
    ticket_id: str
    customer_id: str
    timestamp: datetime
    subject: str
    message: str
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.OPEN
    category: Optional[str] = None
    assigned_agent: Optional[str] = None
    resolution_time: Optional[float] = None  # in minutes
    customer_satisfaction: Optional[int] = None  # 1-5 scale
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)


class SupportTicketSchema(BaseModel):
    """Pydantic schema for support ticket validation."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_id: str = Field(..., description="Customer identifier")
    timestamp: datetime = Field(..., description="Ticket creation timestamp")
    subject: str = Field(..., min_length=1, max_length=200, description="Ticket subject")
    message: str = Field(..., min_length=1, max_length=5000, description="Customer message")
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM, description="Ticket priority")
    status: TicketStatus = Field(default=TicketStatus.OPEN, description="Ticket status")
    category: Optional[str] = Field(None, description="Ticket category")
    assigned_agent: Optional[str] = Field(None, description="Assigned support agent")
    resolution_time: Optional[float] = Field(None, ge=0, description="Resolution time in minutes")
    customer_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Customer satisfaction rating")
    tags: List[str] = Field(default_factory=list, description="Ticket tags")
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict, description="Additional metadata")

    @validator('customer_id')
    def validate_customer_id(cls, v):
        """Validate customer ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Customer ID cannot be empty')
        return v.strip()

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class KnowledgeBaseEntry:
    """Knowledge base entry structure."""
    entry_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    success_rate: float = 0.0
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)


@dataclass
class IntentPrediction:
    """Intent prediction result."""
    intent: IntentType
    confidence: float
    alternative_intents: List[tuple[IntentType, float]] = field(default_factory=list)
    features_used: List[str] = field(default_factory=list)


@dataclass
class AutomatedResponse:
    """Automated response structure."""
    response_id: str
    ticket_id: str
    response_text: str
    response_type: ResponseType
    confidence: float
    intent: IntentType
    knowledge_base_entries: List[str] = field(default_factory=list)
    processing_time: float = 0.0  # in seconds
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)


class CustomerInteraction(BaseModel):
    """Customer interaction tracking."""
    interaction_id: str
    customer_id: str
    timestamp: datetime
    channel: str  # email, chat, phone, etc.
    message: str
    response: Optional[str] = None
    response_time: Optional[float] = None
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=5)
    escalated: bool = False
    resolved: bool = False


def create_synthetic_ticket(
    ticket_id: str,
    customer_id: str,
    subject: str,
    message: str,
    priority: TicketPriority = TicketPriority.MEDIUM,
    **kwargs
) -> SupportTicket:
    """Create a synthetic support ticket."""
    return SupportTicket(
        ticket_id=ticket_id,
        customer_id=customer_id,
        timestamp=datetime.now(),
        subject=subject,
        message=message,
        priority=priority,
        **kwargs
    )


def tickets_to_dataframe(tickets: List[SupportTicket]) -> pd.DataFrame:
    """Convert list of tickets to pandas DataFrame."""
    data = []
    for ticket in tickets:
        data.append({
            'ticket_id': ticket.ticket_id,
            'customer_id': ticket.customer_id,
            'timestamp': ticket.timestamp,
            'subject': ticket.subject,
            'message': ticket.message,
            'priority': ticket.priority.value,
            'status': ticket.status.value,
            'category': ticket.category,
            'assigned_agent': ticket.assigned_agent,
            'resolution_time': ticket.resolution_time,
            'customer_satisfaction': ticket.customer_satisfaction,
            'tags': ','.join(ticket.tags),
            'metadata': str(ticket.metadata)
        })
    return pd.DataFrame(data)


def dataframe_to_tickets(df: pd.DataFrame) -> List[SupportTicket]:
    """Convert pandas DataFrame to list of tickets."""
    tickets = []
    for _, row in df.iterrows():
        ticket = SupportTicket(
            ticket_id=row['ticket_id'],
            customer_id=row['customer_id'],
            timestamp=pd.to_datetime(row['timestamp']),
            subject=row['subject'],
            message=row['message'],
            priority=TicketPriority(row['priority']),
            status=TicketStatus(row['status']),
            category=row.get('category'),
            assigned_agent=row.get('assigned_agent'),
            resolution_time=row.get('resolution_time'),
            customer_satisfaction=row.get('customer_satisfaction'),
            tags=row.get('tags', '').split(',') if row.get('tags') else [],
            metadata=eval(row.get('metadata', '{}')) if row.get('metadata') else {}
        )
        tickets.append(ticket)
    return tickets
