"""
Test suite for customer support automation system.

This module contains comprehensive tests for all components of the system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List

# Import our modules
from src.data.schemas import (
    SupportTicket, TicketPriority, TicketStatus, IntentType,
    KnowledgeBaseEntry, IntentPrediction, AutomatedResponse,
    create_synthetic_ticket, tickets_to_dataframe, dataframe_to_tickets
)
from src.data.generator import SyntheticDataGenerator
from src.models.intent_classifier import (
    KeywordIntentClassifier, MLIntentClassifier, IntentClassifierEnsemble
)
from src.models.response_generator import (
    KnowledgeBaseManager, ResponseGenerator, SupportAutomationSystem
)
from src.eval.metrics import (
    IntentClassificationEvaluator, ResponseGenerationEvaluator,
    BusinessMetricsEvaluator, ComprehensiveEvaluator
)


class TestDataSchemas:
    """Test data schema classes."""
    
    def test_support_ticket_creation(self):
        """Test SupportTicket creation."""
        ticket = SupportTicket(
            ticket_id="TEST_001",
            customer_id="CUSTOMER_001",
            timestamp=datetime.now(),
            subject="Test Subject",
            message="Test message"
        )
        
        assert ticket.ticket_id == "TEST_001"
        assert ticket.customer_id == "CUSTOMER_001"
        assert ticket.subject == "Test Subject"
        assert ticket.message == "Test message"
        assert ticket.priority == TicketPriority.MEDIUM
        assert ticket.status == TicketStatus.OPEN
    
    def test_intent_type_enum(self):
        """Test IntentType enum."""
        assert IntentType.ORDER_STATUS.value == "order_status"
        assert IntentType.RETURN_POLICY.value == "return_policy"
        assert len(IntentType) == 10
    
    def test_create_synthetic_ticket(self):
        """Test synthetic ticket creation."""
        ticket = create_synthetic_ticket(
            ticket_id="SYNTH_001",
            customer_id="CUSTOMER_001",
            subject="Test Subject",
            message="Test message",
            priority=TicketPriority.HIGH
        )
        
        assert ticket.ticket_id == "SYNTH_001"
        assert ticket.priority == TicketPriority.HIGH
        assert isinstance(ticket.timestamp, datetime)
    
    def test_tickets_dataframe_conversion(self):
        """Test conversion between tickets and DataFrame."""
        tickets = [
            create_synthetic_ticket("T1", "C1", "S1", "M1"),
            create_synthetic_ticket("T2", "C2", "S2", "M2")
        ]
        
        df = tickets_to_dataframe(tickets)
        assert len(df) == 2
        assert "ticket_id" in df.columns
        assert "customer_id" in df.columns
        
        converted_tickets = dataframe_to_tickets(df)
        assert len(converted_tickets) == 2
        assert converted_tickets[0].ticket_id == "T1"


class TestDataGenerator:
    """Test synthetic data generation."""
    
    def test_synthetic_data_generator(self):
        """Test SyntheticDataGenerator."""
        generator = SyntheticDataGenerator(seed=42)
        
        # Test ticket generation
        tickets = generator.generate_tickets(10)
        assert len(tickets) == 10
        assert all(isinstance(ticket, SupportTicket) for ticket in tickets)
        
        # Test knowledge base generation
        kb_entries = generator.generate_knowledge_base()
        assert len(kb_entries) > 0
        assert all(isinstance(entry, KnowledgeBaseEntry) for entry in kb_entries)
        
        # Test interaction generation
        interactions = generator.generate_customer_interactions(5)
        assert len(interactions) == 5
        assert all("interaction_id" in interaction for interaction in interactions)
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        generator1 = SyntheticDataGenerator(seed=42)
        generator2 = SyntheticDataGenerator(seed=42)
        
        tickets1 = generator1.generate_tickets(10)
        tickets2 = generator2.generate_tickets(10)
        
        assert tickets1[0].ticket_id == tickets2[0].ticket_id
        assert tickets1[0].message == tickets2[0].message


class TestIntentClassifier:
    """Test intent classification models."""
    
    def test_keyword_classifier(self):
        """Test KeywordIntentClassifier."""
        classifier = KeywordIntentClassifier()
        
        # Test order status intent
        prediction = classifier.predict("What's the status of my order?")
        assert prediction.intent == IntentType.ORDER_STATUS
        assert prediction.confidence > 0
        
        # Test return policy intent
        prediction = classifier.predict("I want to return this item")
        assert prediction.intent == IntentType.RETURN_POLICY
        assert prediction.confidence > 0
        
        # Test unknown intent
        prediction = classifier.predict("Random gibberish text")
        assert prediction.intent == IntentType.GENERAL_INQUIRY
        assert prediction.confidence < 0.5
    
    def test_ml_classifier(self):
        """Test MLIntentClassifier."""
        classifier = MLIntentClassifier()
        
        # Generate training data
        generator = SyntheticDataGenerator(seed=42)
        tickets = generator.generate_tickets(50)
        
        # Train classifier
        results = classifier.train(tickets)
        assert "accuracy" in results
        assert results["accuracy"] >= 0
        
        # Test prediction
        prediction = classifier.predict("What's my order status?")
        assert isinstance(prediction, IntentPrediction)
        assert prediction.confidence >= 0
        assert prediction.confidence <= 1
    
    def test_ensemble_classifier(self):
        """Test IntentClassifierEnsemble."""
        classifier = IntentClassifierEnsemble()
        
        # Generate training data
        generator = SyntheticDataGenerator(seed=42)
        tickets = generator.generate_tickets(50)
        
        # Train classifier
        results = classifier.train(tickets)
        assert isinstance(results, dict)
        
        # Test prediction
        prediction = classifier.predict("I need help with my order")
        assert isinstance(prediction, IntentPrediction)
        assert prediction.confidence >= 0
        assert prediction.confidence <= 1


class TestResponseGenerator:
    """Test response generation components."""
    
    def test_knowledge_base_manager(self):
        """Test KnowledgeBaseManager."""
        kb_manager = KnowledgeBaseManager()
        
        # Create test entries
        entries = [
            KnowledgeBaseEntry(
                entry_id="KB_001",
                title="Order Status Help",
                content="You can check your order status online",
                category="order_status",
                tags=["order", "status"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            KnowledgeBaseEntry(
                entry_id="KB_002",
                title="Return Policy",
                content="Returns are allowed within 30 days",
                category="return_policy",
                tags=["return", "policy"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        kb_manager.add_entries(entries)
        assert len(kb_manager.entries) == 2
        
        # Test search
        results = kb_manager.search_similar("order status", top_k=1)
        assert len(results) > 0
        assert results[0][1] > 0  # similarity score
        
        # Test search by intent
        intent_results = kb_manager.search_by_intent(IntentType.ORDER_STATUS)
        assert len(intent_results) > 0
    
    def test_response_generator(self):
        """Test ResponseGenerator."""
        kb_manager = KnowledgeBaseManager()
        response_generator = ResponseGenerator(kb_manager)
        
        # Create test ticket
        ticket = create_synthetic_ticket(
            "TEST_001", "CUSTOMER_001", "Test", "What's my order status?"
        )
        
        # Create test intent prediction
        intent_prediction = IntentPrediction(
            intent=IntentType.ORDER_STATUS,
            confidence=0.8
        )
        
        # Generate response
        response = response_generator.generate_response(ticket, intent_prediction)
        
        assert isinstance(response, AutomatedResponse)
        assert response.ticket_id == "TEST_001"
        assert response.intent == IntentType.ORDER_STATUS
        assert response.confidence == 0.8
        assert len(response.response_text) > 0
    
    def test_support_automation_system(self):
        """Test SupportAutomationSystem."""
        # Initialize components
        kb_manager = KnowledgeBaseManager()
        intent_classifier = KeywordIntentClassifier()
        
        system = SupportAutomationSystem(intent_classifier, kb_manager)
        
        # Create test ticket
        ticket = create_synthetic_ticket(
            "TEST_001", "CUSTOMER_001", "Test", "What's my order status?"
        )
        
        # Process ticket
        response = system.process_ticket(ticket)
        
        assert isinstance(response, AutomatedResponse)
        assert response.ticket_id == "TEST_001"
        assert len(response.response_text) > 0


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_intent_classification_evaluator(self):
        """Test IntentClassificationEvaluator."""
        evaluator = IntentClassificationEvaluator()
        
        # Add test predictions
        evaluator.add_prediction(
            IntentPrediction(IntentType.ORDER_STATUS, 0.9),
            IntentType.ORDER_STATUS,
            0.5
        )
        evaluator.add_prediction(
            IntentPrediction(IntentType.RETURN_POLICY, 0.8),
            IntentType.RETURN_POLICY,
            0.3
        )
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics()
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert metrics["accuracy"] == 1.0  # Both predictions correct
    
    def test_response_generation_evaluator(self):
        """Test ResponseGenerationEvaluator."""
        evaluator = ResponseGenerationEvaluator()
        
        # Add test responses
        response1 = AutomatedResponse(
            response_id="R1",
            ticket_id="T1",
            response_text="Test response 1",
            response_type="automated",
            confidence=0.8,
            intent=IntentType.ORDER_STATUS
        )
        
        response2 = AutomatedResponse(
            response_id="R2",
            ticket_id="T2",
            response_text="Test response 2",
            response_type="escalated",
            confidence=0.3,
            intent=IntentType.GENERAL_INQUIRY
        )
        
        evaluator.add_response(response1)
        evaluator.add_response(response2)
        
        # Add feedback
        evaluator.add_feedback("R1", 4, True, False)
        evaluator.add_feedback("R2", 2, False, True)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics()
        assert "total_responses" in metrics
        assert "automation_rate" in metrics
        assert "escalation_rate" in metrics
        assert metrics["total_responses"] == 2
        assert metrics["automation_rate"] == 0.5
    
    def test_business_metrics_evaluator(self):
        """Test BusinessMetricsEvaluator."""
        evaluator = BusinessMetricsEvaluator()
        
        # Add test tickets
        ticket1 = create_synthetic_ticket(
            "T1", "C1", "Test 1", "Message 1",
            resolution_time=30.0,
            customer_satisfaction=4
        )
        ticket2 = create_synthetic_ticket(
            "T2", "C2", "Test 2", "Message 2",
            resolution_time=60.0,
            customer_satisfaction=3
        )
        
        evaluator.add_ticket(ticket1)
        evaluator.add_ticket(ticket2)
        
        # Add test responses
        response1 = AutomatedResponse(
            response_id="R1", ticket_id="T1", response_text="R1",
            response_type="automated", confidence=0.8, intent=IntentType.ORDER_STATUS
        )
        response2 = AutomatedResponse(
            response_id="R2", ticket_id="T2", response_text="R2",
            response_type="escalated", confidence=0.3, intent=IntentType.GENERAL_INQUIRY
        )
        
        evaluator.add_response(response1)
        evaluator.add_response(response2)
        
        # Calculate metrics
        cost_metrics = evaluator.calculate_cost_savings()
        service_metrics = evaluator.calculate_service_level_metrics()
        
        assert "cost_savings" in cost_metrics
        assert "total_tickets" in service_metrics
        assert service_metrics["total_tickets"] == 2
        assert service_metrics["avg_customer_satisfaction"] == 3.5


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        generator = SyntheticDataGenerator(seed=42)
        tickets = generator.generate_tickets(20)
        kb_entries = generator.generate_knowledge_base()
        
        # Initialize system
        kb_manager = KnowledgeBaseManager()
        kb_manager.add_entries(kb_entries)
        
        intent_classifier = KeywordIntentClassifier()
        system = SupportAutomationSystem(intent_classifier, kb_manager)
        
        # Process tickets
        responses = []
        for ticket in tickets[:5]:  # Process first 5 tickets
            response = system.process_ticket(ticket)
            responses.append(response)
        
        assert len(responses) == 5
        assert all(isinstance(r, AutomatedResponse) for r in responses)
        
        # Evaluate system
        evaluator = ComprehensiveEvaluator()
        metrics = evaluator.evaluate_system(tickets[:5], responses)
        
        assert "intent_classification" in metrics
        assert "response_generation" in metrics
        assert "cost_analysis" in metrics
        assert "service_level" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
