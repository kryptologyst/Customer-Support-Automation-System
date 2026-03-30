"""
Data generation utilities for creating synthetic customer support datasets.

This module provides functions to generate realistic synthetic data for
training and testing the customer support automation system.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .schemas import (
    SupportTicket, TicketPriority, TicketStatus, IntentType,
    KnowledgeBaseEntry, create_synthetic_ticket
)


class SyntheticDataGenerator:
    """Generator for synthetic customer support data."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Sample data templates
        self.customer_ids = [f"CUST_{i:06d}" for i in range(1000, 2000)]
        self.agent_ids = [f"AGENT_{i:03d}" for i in range(1, 21)]
        
        # Intent-specific message templates
        self.intent_templates = {
            IntentType.ORDER_STATUS: [
                "What's the status of my order?",
                "Can you check my order status?",
                "I need to know where my order is",
                "When will my order arrive?",
                "Order tracking information please"
            ],
            IntentType.RETURN_POLICY: [
                "What's your return policy?",
                "How do I return an item?",
                "Can I get a refund?",
                "Return process information",
                "I want to return this product"
            ],
            IntentType.CANCEL_ORDER: [
                "I need to cancel my order",
                "How do I cancel an order?",
                "Cancel my recent purchase",
                "I want to cancel this order",
                "Order cancellation process"
            ],
            IntentType.PAYMENT_FAILED: [
                "My payment failed",
                "Payment error occurred",
                "Card was declined",
                "Payment processing issue",
                "Transaction failed"
            ],
            IntentType.DELIVERY_TIME: [
                "When will this be delivered?",
                "Delivery time estimate",
                "Shipping duration",
                "How long for delivery?",
                "Expected delivery date"
            ],
            IntentType.PRODUCT_INFO: [
                "Tell me about this product",
                "Product specifications",
                "What are the features?",
                "Product details please",
                "More information about item"
            ],
            IntentType.TECHNICAL_SUPPORT: [
                "I'm having technical issues",
                "Product not working properly",
                "Need technical help",
                "Device malfunction",
                "Software problem"
            ],
            IntentType.BILLING_QUESTION: [
                "Billing inquiry",
                "Question about my bill",
                "Charged incorrectly",
                "Payment method issue",
                "Invoice question"
            ],
            IntentType.ACCOUNT_ISSUE: [
                "Can't access my account",
                "Login problems",
                "Account locked",
                "Password reset needed",
                "Account access issue"
            ],
            IntentType.GENERAL_INQUIRY: [
                "General question",
                "Need help with something",
                "Customer service inquiry",
                "Have a question",
                "Need assistance"
            ]
        }
        
        # Knowledge base entries
        self.knowledge_base_templates = {
            IntentType.ORDER_STATUS: [
                "You can check your order status by logging into your account and visiting 'My Orders'.",
                "Order tracking information is available in your account dashboard.",
                "You'll receive email updates about your order status.",
                "Use the order number to track your package on our website."
            ],
            IntentType.RETURN_POLICY: [
                "Our return policy allows returns within 30 days of delivery.",
                "Items must be in original condition with tags attached.",
                "Return shipping is free for eligible items.",
                "Refunds are processed within 5-7 business days."
            ],
            IntentType.CANCEL_ORDER: [
                "To cancel an order, go to 'My Orders', select the order, and click 'Cancel'.",
                "Orders can be cancelled within 1 hour of placement.",
                "Cancelled orders are refunded within 24 hours.",
                "Contact support if cancellation option is not available."
            ],
            IntentType.PAYMENT_FAILED: [
                "Please check your payment method or try using a different card.",
                "Ensure your billing address matches your card information.",
                "Contact your bank if the issue persists.",
                "Try using a different payment method."
            ],
            IntentType.DELIVERY_TIME: [
                "Delivery usually takes 3-5 business days depending on your location.",
                "Express shipping options are available for faster delivery.",
                "Delivery times may vary during peak seasons.",
                "You'll receive tracking information once your order ships."
            ]
        }

    def generate_tickets(self, n_tickets: int = 1000) -> List[SupportTicket]:
        """Generate synthetic support tickets."""
        tickets = []
        
        for i in range(n_tickets):
            # Random intent selection
            intent = random.choice(list(IntentType))
            
            # Generate ticket data
            ticket_id = f"TICKET_{i+1:06d}"
            customer_id = random.choice(self.customer_ids)
            
            # Generate timestamp (last 90 days)
            days_ago = random.randint(0, 90)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            timestamp = datetime.now() - timedelta(
                days=days_ago, hours=hours_ago, minutes=minutes_ago
            )
            
            # Generate subject and message
            subject = f"Re: {intent.value.replace('_', ' ').title()}"
            message = random.choice(self.intent_templates[intent])
            
            # Add some variation to messages
            if random.random() < 0.3:
                message += f" Order ID: {random.randint(100000, 999999)}"
            
            # Priority distribution
            priority_weights = [0.1, 0.6, 0.25, 0.05]  # low, medium, high, urgent
            priority = random.choices(
                list(TicketPriority), weights=priority_weights
            )[0]
            
            # Status distribution
            status_weights = [0.3, 0.2, 0.4, 0.08, 0.02]  # open, in_progress, resolved, closed, escalated
            status = random.choices(
                list(TicketStatus), weights=status_weights
            )[0]
            
            # Resolution time (if resolved)
            resolution_time = None
            if status in [TicketStatus.RESOLVED, TicketStatus.CLOSED]:
                resolution_time = random.uniform(5, 120)  # 5 minutes to 2 hours
            
            # Customer satisfaction (if resolved)
            customer_satisfaction = None
            if status in [TicketStatus.RESOLVED, TicketStatus.CLOSED]:
                # Higher satisfaction for automated responses
                satisfaction_weights = [0.05, 0.1, 0.2, 0.4, 0.25]  # 1-5 scale
                customer_satisfaction = random.choices(
                    range(1, 6), weights=satisfaction_weights
                )[0]
            
            # Generate tags
            tags = [intent.value]
            if priority == TicketPriority.URGENT:
                tags.append("urgent")
            if random.random() < 0.2:
                tags.append("repeat_customer")
            
            ticket = SupportTicket(
                ticket_id=ticket_id,
                customer_id=customer_id,
                timestamp=timestamp,
                subject=subject,
                message=message,
                priority=priority,
                status=status,
                category=intent.value,
                assigned_agent=random.choice(self.agent_ids) if status != TicketStatus.OPEN else None,
                resolution_time=resolution_time,
                customer_satisfaction=customer_satisfaction,
                tags=tags,
                metadata={
                    "intent": intent.value,
                    "channel": random.choice(["email", "chat", "phone", "web"]),
                    "device": random.choice(["desktop", "mobile", "tablet"]),
                    "browser": random.choice(["chrome", "firefox", "safari", "edge"])
                }
            )
            tickets.append(ticket)
        
        return tickets

    def generate_knowledge_base(self) -> List[KnowledgeBaseEntry]:
        """Generate synthetic knowledge base entries."""
        entries = []
        
        for intent, responses in self.knowledge_base_templates.items():
            for i, response in enumerate(responses):
                entry = KnowledgeBaseEntry(
                    entry_id=f"KB_{intent.value}_{i+1:03d}",
                    title=f"{intent.value.replace('_', ' ').title()} - Response {i+1}",
                    content=response,
                    category=intent.value,
                    tags=[intent.value],
                    created_at=datetime.now() - timedelta(days=random.randint(30, 365)),
                    updated_at=datetime.now() - timedelta(days=random.randint(1, 30)),
                    usage_count=random.randint(10, 500),
                    success_rate=random.uniform(0.7, 0.95),
                    metadata={
                        "intent": intent.value,
                        "confidence": random.uniform(0.8, 0.95),
                        "last_used": datetime.now() - timedelta(days=random.randint(1, 7))
                    }
                )
                entries.append(entry)
        
        return entries

    def generate_customer_interactions(self, n_interactions: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic customer interaction data."""
        interactions = []
        
        for i in range(n_interactions):
            intent = random.choice(list(IntentType))
            customer_id = random.choice(self.customer_ids)
            
            interaction = {
                "interaction_id": f"INT_{i+1:06d}",
                "customer_id": customer_id,
                "timestamp": datetime.now() - timedelta(
                    days=random.randint(0, 30),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                ),
                "channel": random.choice(["email", "chat", "phone", "web", "mobile_app"]),
                "message": random.choice(self.intent_templates[intent]),
                "intent": intent.value,
                "response_time": random.uniform(0.5, 10.0),  # seconds
                "satisfaction_rating": random.randint(1, 5),
                "escalated": random.random() < 0.1,
                "resolved": random.random() < 0.8,
                "automated_response": random.random() < 0.6
            }
            interactions.append(interaction)
        
        return interactions

    def save_synthetic_data(self, output_dir: str = "data/synthetic"):
        """Generate and save all synthetic data."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data
        tickets = self.generate_tickets(1000)
        knowledge_base = self.generate_knowledge_base()
        interactions = self.generate_customer_interactions(500)
        
        # Convert to DataFrames and save
        tickets_df = pd.DataFrame([
            {
                'ticket_id': t.ticket_id,
                'customer_id': t.customer_id,
                'timestamp': t.timestamp,
                'subject': t.subject,
                'message': t.message,
                'priority': t.priority.value,
                'status': t.status.value,
                'category': t.category,
                'assigned_agent': t.assigned_agent,
                'resolution_time': t.resolution_time,
                'customer_satisfaction': t.customer_satisfaction,
                'tags': ','.join(t.tags),
                'metadata': str(t.metadata)
            }
            for t in tickets
        ])
        
        kb_df = pd.DataFrame([
            {
                'entry_id': kb.entry_id,
                'title': kb.title,
                'content': kb.content,
                'category': kb.category,
                'tags': ','.join(kb.tags),
                'created_at': kb.created_at,
                'updated_at': kb.updated_at,
                'usage_count': kb.usage_count,
                'success_rate': kb.success_rate,
                'metadata': str(kb.metadata)
            }
            for kb in knowledge_base
        ])
        
        interactions_df = pd.DataFrame(interactions)
        
        # Save to CSV
        tickets_df.to_csv(f"{output_dir}/support_tickets.csv", index=False)
        kb_df.to_csv(f"{output_dir}/knowledge_base.csv", index=False)
        interactions_df.to_csv(f"{output_dir}/customer_interactions.csv", index=False)
        
        print(f"Generated synthetic data saved to {output_dir}/")
        print(f"- {len(tickets)} support tickets")
        print(f"- {len(knowledge_base)} knowledge base entries")
        print(f"- {len(interactions)} customer interactions")
        
        return tickets_df, kb_df, interactions_df


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.save_synthetic_data()
