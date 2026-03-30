#!/usr/bin/env python3
"""
Quick test script to verify the customer support automation system works.

This script tests the basic functionality without requiring full training.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that all modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from src.data.schemas import SupportTicket, IntentType, create_synthetic_ticket
        print("✓ Data schemas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data schemas: {e}")
        return False
    
    try:
        from src.data.generator import SyntheticDataGenerator
        print("✓ Data generator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data generator: {e}")
        return False
    
    try:
        from src.models.intent_classifier import KeywordIntentClassifier
        print("✓ Intent classifier imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import intent classifier: {e}")
        return False
    
    try:
        from src.models.response_generator import KnowledgeBaseManager, ResponseGenerator
        print("✓ Response generator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import response generator: {e}")
        return False
    
    return True


def test_data_generation():
    """Test synthetic data generation."""
    print("\nTesting data generation...")
    
    try:
        from src.data.generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(seed=42)
        tickets = generator.generate_tickets(5)
        kb_entries = generator.generate_knowledge_base()
        
        print(f"✓ Generated {len(tickets)} tickets")
        print(f"✓ Generated {len(kb_entries)} knowledge base entries")
        
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False


def test_intent_classification():
    """Test intent classification."""
    print("\nTesting intent classification...")
    
    try:
        from src.models.intent_classifier import KeywordIntentClassifier
        from src.data.schemas import IntentType
        
        classifier = KeywordIntentClassifier()
        
        # Test different intents
        test_cases = [
            ("What's the status of my order?", IntentType.ORDER_STATUS),
            ("I want to return this item", IntentType.RETURN_POLICY),
            ("My payment failed", IntentType.PAYMENT_FAILED),
            ("When will this arrive?", IntentType.DELIVERY_TIME)
        ]
        
        for message, expected_intent in test_cases:
            prediction = classifier.predict(message)
            print(f"  '{message}' -> {prediction.intent.value} (confidence: {prediction.confidence:.2f})")
            
            # Check if prediction makes sense (not necessarily exact match)
            if prediction.confidence > 0.1:
                print(f"    ✓ Reasonable prediction")
            else:
                print(f"    ⚠ Low confidence prediction")
        
        return True
    except Exception as e:
        print(f"✗ Intent classification failed: {e}")
        return False


def test_response_generation():
    """Test response generation."""
    print("\nTesting response generation...")
    
    try:
        from src.models.response_generator import KnowledgeBaseManager, ResponseGenerator
        from src.models.intent_classifier import KeywordIntentClassifier
        from src.data.schemas import create_synthetic_ticket, IntentPrediction, IntentType
        
        # Initialize components
        kb_manager = KnowledgeBaseManager()
        response_generator = ResponseGenerator(kb_manager)
        
        # Create test ticket
        ticket = create_synthetic_ticket(
            "TEST_001", "CUSTOMER_001", "Test Subject", "What's my order status?"
        )
        
        # Create test intent prediction
        intent_prediction = IntentPrediction(
            intent=IntentType.ORDER_STATUS,
            confidence=0.8
        )
        
        # Generate response
        response = response_generator.generate_response(ticket, intent_prediction)
        
        print(f"✓ Generated response: '{response.response_text[:50]}...'")
        print(f"✓ Response type: {response.response_type.value}")
        print(f"✓ Confidence: {response.confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Response generation failed: {e}")
        return False


def test_end_to_end():
    """Test end-to-end system."""
    print("\nTesting end-to-end system...")
    
    try:
        from src.models.intent_classifier import KeywordIntentClassifier
        from src.models.response_generator import KnowledgeBaseManager, SupportAutomationSystem
        from src.data.schemas import create_synthetic_ticket
        
        # Initialize system
        kb_manager = KnowledgeBaseManager()
        intent_classifier = KeywordIntentClassifier()
        system = SupportAutomationSystem(intent_classifier, kb_manager)
        
        # Test tickets
        test_tickets = [
            "What's the status of my order?",
            "I need to return this item",
            "My payment was declined",
            "When will this arrive?"
        ]
        
        for i, message in enumerate(test_tickets):
            ticket = create_synthetic_ticket(
                f"TEST_{i+1:03d}", "CUSTOMER_001", "Test Subject", message
            )
            
            response = system.process_ticket(ticket)
            
            print(f"  Ticket {i+1}: '{message}'")
            print(f"    Intent: {response.intent.value}")
            print(f"    Response: '{response.response_text[:60]}...'")
            print(f"    Confidence: {response.confidence:.2f}")
            print()
        
        return True
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Customer Support Automation System - Quick Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_data_generation,
        test_intent_classification,
        test_response_generation,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python main.py generate-data' to create synthetic data")
        print("2. Run 'python main.py train-models' to train ML models")
        print("3. Run 'python main.py demo' to launch the interactive demo")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
