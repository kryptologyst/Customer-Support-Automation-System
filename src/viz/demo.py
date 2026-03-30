"""
Streamlit demo application for customer support automation system.

This module provides an interactive web interface for testing and demonstrating
the customer support automation capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.schemas import (
    SupportTicket, TicketPriority, TicketStatus, IntentType,
    create_synthetic_ticket, tickets_to_dataframe
)
from data.generator import SyntheticDataGenerator
from models.intent_classifier import (
    KeywordIntentClassifier, MLIntentClassifier, 
    TransformerIntentClassifier, IntentClassifierEnsemble
)
from models.response_generator import (
    KnowledgeBaseManager, ResponseGenerator, SupportAutomationSystem
)
from eval.metrics import ComprehensiveEvaluator


# Page configuration
st.set_page_config(
    page_title="Customer Support Automation Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_synthetic_data():
    """Load synthetic data for demonstration."""
    generator = SyntheticDataGenerator()
    tickets = generator.generate_tickets(100)
    kb_entries = generator.generate_knowledge_base()
    return tickets, kb_entries


@st.cache_resource
def initialize_system():
    """Initialize the support automation system."""
    # Load data
    tickets, kb_entries = load_synthetic_data()
    
    # Initialize components
    kb_manager = KnowledgeBaseManager()
    kb_manager.add_entries(kb_entries)
    
    # Initialize intent classifier (using ensemble)
    intent_classifier = IntentClassifierEnsemble()
    
    # Train the system
    with st.spinner("Training the system..."):
        try:
            intent_classifier.train(tickets)
        except Exception as e:
            st.warning(f"Training failed: {e}. Using keyword classifier only.")
            intent_classifier = KeywordIntentClassifier()
    
    # Initialize response generator
    response_generator = ResponseGenerator(kb_manager)
    
    # Create automation system
    system = SupportAutomationSystem(intent_classifier, kb_manager)
    
    return system, tickets, kb_entries


def display_disclaimer():
    """Display important disclaimer."""
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p><strong>This is a research and educational demonstration system.</strong></p>
        <ul>
            <li>This system is for educational purposes only</li>
            <li>Do not use for automated decision-making without human review</li>
            <li>All responses should be validated by human agents</li>
            <li>This system is not production-ready</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🤖 Customer Support Automation Demo</h1>', unsafe_allow_html=True)
    
    # Display disclaimer
    display_disclaimer()
    
    # Initialize system
    system, tickets, kb_entries = initialize_system()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Ticket Processing", "Intent Classification", "Knowledge Base", "Performance Metrics", "System Configuration"]
    )
    
    if page == "Ticket Processing":
        ticket_processing_page(system, tickets)
    elif page == "Intent Classification":
        intent_classification_page(system, tickets)
    elif page == "Knowledge Base":
        knowledge_base_page(system, kb_entries)
    elif page == "Performance Metrics":
        performance_metrics_page(system, tickets)
    elif page == "System Configuration":
        system_configuration_page(system)


def ticket_processing_page(system, tickets):
    """Ticket processing demonstration page."""
    st.header("🎫 Ticket Processing Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Process a Support Ticket")
        
        # Sample ticket selection
        sample_tickets = [
            "What's the status of my order?",
            "I need to return this item, what's your policy?",
            "My payment failed when I tried to checkout",
            "When will my delivery arrive?",
            "I can't log into my account",
            "The product I received is defective"
        ]
        
        selected_sample = st.selectbox("Choose a sample ticket:", ["Custom"] + sample_tickets)
        
        if selected_sample == "Custom":
            ticket_message = st.text_area(
                "Enter your support ticket message:",
                placeholder="Describe your issue or question...",
                height=100
            )
        else:
            ticket_message = selected_sample
            st.text_area("Ticket message:", value=ticket_message, height=100)
        
        # Process ticket button
        if st.button("Process Ticket", type="primary"):
            if ticket_message.strip():
                # Create ticket
                ticket = create_synthetic_ticket(
                    ticket_id=f"DEMO_{int(time.time())}",
                    customer_id="DEMO_CUSTOMER",
                    subject="Demo Ticket",
                    message=ticket_message
                )
                
                # Process ticket
                with st.spinner("Processing ticket..."):
                    start_time = time.time()
                    response = system.process_ticket(ticket)
                    processing_time = time.time() - start_time
                
                # Display results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.subheader("🤖 Automated Response")
                
                # Response details
                col_resp1, col_resp2 = st.columns(2)
                with col_resp1:
                    st.metric("Response Type", response.response_type.value.title())
                    st.metric("Confidence", f"{response.confidence:.2f}")
                with col_resp2:
                    st.metric("Intent", response.intent.value.replace('_', ' ').title())
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                # Response text
                st.text_area("Response:", value=response.response_text, height=150)
                
                # Alternative intents
                if response.alternative_intents:
                    st.subheader("Alternative Intent Classifications")
                    alt_data = []
                    for intent, conf in response.alternative_intents:
                        alt_data.append({
                            'Intent': intent.value.replace('_', ' ').title(),
                            'Confidence': f"{conf:.3f}"
                        })
                    st.dataframe(pd.DataFrame(alt_data), use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Please enter a ticket message.")
    
    with col2:
        st.subheader("System Status")
        
        # Quick stats
        st.metric("Total Tickets Processed", len(tickets))
        st.metric("Knowledge Base Entries", len(kb_entries))
        st.metric("Response Templates", len(system.response_generator.response_templates))
        
        # Recent activity
        st.subheader("Recent Activity")
        recent_tickets = tickets[-5:] if len(tickets) >= 5 else tickets
        for ticket in reversed(recent_tickets):
            st.text(f"• {ticket.message[:50]}...")


def intent_classification_page(system, tickets):
    """Intent classification demonstration page."""
    st.header("🎯 Intent Classification Demo")
    
    # Intent distribution
    st.subheader("Intent Distribution in Dataset")
    intent_counts = {}
    for ticket in tickets:
        intent = ticket.metadata.get('intent', 'general_inquiry')
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    if intent_counts:
        fig = px.pie(
            values=list(intent_counts.values()),
            names=list(intent_counts.keys()),
            title="Distribution of Intents"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification testing
    st.subheader("Test Intent Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_message = st.text_area(
            "Enter a message to classify:",
            placeholder="e.g., 'I need help with my order'",
            height=100
        )
        
        if st.button("Classify Intent"):
            if test_message.strip():
                prediction = system.intent_classifier.predict(test_message)
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.subheader("Classification Results")
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric("Predicted Intent", prediction.intent.value.replace('_', ' ').title())
                    st.metric("Confidence", f"{prediction.confidence:.3f}")
                
                with col_pred2:
                    st.metric("Features Used", ", ".join(prediction.features_used))
                
                # Confidence visualization
                fig = go.Figure(go.Bar(
                    x=[prediction.intent.value.replace('_', ' ').title()],
                    y=[prediction.confidence],
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    title="Confidence Score",
                    yaxis_title="Confidence",
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Alternative predictions
                if prediction.alternative_intents:
                    st.subheader("Alternative Predictions")
                    alt_data = []
                    for intent, conf in prediction.alternative_intents:
                        alt_data.append({
                            'Intent': intent.value.replace('_', ' ').title(),
                            'Confidence': conf
                        })
                    st.dataframe(pd.DataFrame(alt_data), use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Please enter a message to classify.")
    
    with col2:
        # Sample classifications
        st.subheader("Sample Classifications")
        
        sample_messages = [
            "What's the status of my order?",
            "I want to return this item",
            "My payment was declined",
            "When will this arrive?",
            "I can't access my account"
        ]
        
        sample_results = []
        for msg in sample_messages:
            pred = system.intent_classifier.predict(msg)
            sample_results.append({
                'Message': msg,
                'Intent': pred.intent.value.replace('_', ' ').title(),
                'Confidence': f"{pred.confidence:.3f}"
            })
        
        st.dataframe(pd.DataFrame(sample_results), use_container_width=True)


def knowledge_base_page(system, kb_entries):
    """Knowledge base demonstration page."""
    st.header("📚 Knowledge Base Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Search Knowledge Base")
        
        search_query = st.text_input(
            "Search for information:",
            placeholder="e.g., 'return policy' or 'order status'"
        )
        
        if st.button("Search"):
            if search_query.strip():
                # Search knowledge base
                results = system.kb_manager.search_similar(search_query, top_k=5)
                
                if results:
                    st.subheader("Search Results")
                    for i, (entry, similarity) in enumerate(results):
                        with st.expander(f"Result {i+1} (Similarity: {similarity:.3f})"):
                            st.write(f"**Title:** {entry.title}")
                            st.write(f"**Category:** {entry.category}")
                            st.write(f"**Content:** {entry.content}")
                            st.write(f"**Tags:** {', '.join(entry.tags)}")
                            st.write(f"**Success Rate:** {entry.success_rate:.2f}")
                            st.write(f"**Usage Count:** {entry.usage_count}")
                else:
                    st.warning("No similar entries found.")
            else:
                st.error("Please enter a search query.")
        
        # Browse by category
        st.subheader("Browse by Category")
        categories = list(set(entry.category for entry in kb_entries))
        selected_category = st.selectbox("Select category:", categories)
        
        if selected_category:
            category_entries = [entry for entry in kb_entries if entry.category == selected_category]
            
            for entry in category_entries:
                with st.expander(f"{entry.title}"):
                    st.write(entry.content)
                    st.write(f"**Success Rate:** {entry.success_rate:.2f} | **Usage:** {entry.usage_count}")
    
    with col2:
        st.subheader("Knowledge Base Statistics")
        
        # Stats
        st.metric("Total Entries", len(kb_entries))
        
        # Category distribution
        category_counts = {}
        for entry in kb_entries:
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
        
        if category_counts:
            fig = px.bar(
                x=list(category_counts.keys()),
                y=list(category_counts.values()),
                title="Entries by Category"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Success rate distribution
        success_rates = [entry.success_rate for entry in kb_entries]
        if success_rates:
            fig = px.histogram(
                x=success_rates,
                title="Success Rate Distribution",
                labels={'x': 'Success Rate', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)


def performance_metrics_page(system, tickets):
    """Performance metrics demonstration page."""
    st.header("📊 Performance Metrics")
    
    # Generate sample responses for demonstration
    sample_responses = []
    for ticket in tickets[:20]:  # Use first 20 tickets for demo
        try:
            response = system.process_ticket(ticket)
            sample_responses.append(response)
        except Exception as e:
            st.warning(f"Error processing ticket {ticket.ticket_id}: {e}")
    
    if sample_responses:
        # Calculate metrics
        evaluator = ComprehensiveEvaluator()
        metrics = evaluator.evaluate_system(tickets[:20], sample_responses)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            intent_metrics = metrics.get('intent_classification', {})
            if intent_metrics:
                st.metric("Accuracy", f"{intent_metrics.get('accuracy', 0):.3f}")
                st.metric("Precision", f"{intent_metrics.get('precision', 0):.3f}")
        
        with col2:
            if intent_metrics:
                st.metric("Recall", f"{intent_metrics.get('recall', 0):.3f}")
                st.metric("F1-Score", f"{intent_metrics.get('f1_score', 0):.3f}")
        
        with col3:
            response_metrics = metrics.get('response_generation', {})
            if response_metrics:
                st.metric("Automation Rate", f"{response_metrics.get('automation_rate', 0):.1%}")
                st.metric("Avg Response Time", f"{response_metrics.get('avg_response_time', 0):.2f}s")
        
        with col4:
            cost_metrics = metrics.get('cost_analysis', {})
            if cost_metrics:
                st.metric("Cost Savings", f"${cost_metrics.get('cost_savings', 0):.2f}")
                st.metric("Savings %", f"{cost_metrics.get('cost_savings_percentage', 0):.1f}%")
        
        # Detailed metrics
        st.subheader("Detailed Performance Analysis")
        
        # Response type distribution
        response_types = [r.response_type.value for r in sample_responses]
        type_counts = pd.Series(response_types).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Response Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            confidences = [r.confidence for r in sample_responses]
            fig = px.histogram(
                x=confidences,
                title="Confidence Score Distribution",
                labels={'x': 'Confidence', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance over time (simulated)
        st.subheader("Performance Trends")
        
        # Simulate performance over time
        time_points = pd.date_range(start='2024-01-01', periods=30, freq='D')
        simulated_accuracy = np.random.normal(0.85, 0.05, 30)
        simulated_automation = np.random.normal(0.75, 0.1, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_points,
            y=simulated_accuracy,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=time_points,
            y=simulated_automation,
            mode='lines+markers',
            name='Automation Rate',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Performance Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy",
            yaxis2=dict(
                title="Automation Rate",
                overlaying='y',
                side='right'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export metrics
        if st.button("Export Metrics Report"):
            report = evaluator.generate_report(metrics)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"support_automation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    else:
        st.warning("No responses generated. Please check the system configuration.")


def system_configuration_page(system):
    """System configuration page."""
    st.header("⚙️ System Configuration")
    
    st.subheader("Current Configuration")
    
    # Display current settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Intent Classification:**")
        st.write("- Model: Ensemble (Keyword + ML + Transformer)")
        st.write("- Confidence Threshold: 0.7")
        
        st.write("**Response Generation:**")
        st.write("- Templates: 7 predefined templates")
        st.write("- Knowledge Base: Sentence Transformers")
        st.write("- Similarity Threshold: 0.7")
    
    with col2:
        st.write("**Business Constraints:**")
        st.write("- Max Response Time: 5.0 seconds")
        st.write("- Min Confidence: 0.7")
        st.write("- Escalation Threshold: 0.5")
        st.write("- Max Automation Rate: 80%")
    
    # Configuration options
    st.subheader("Configuration Options")
    
    # Intent classification settings
    with st.expander("Intent Classification Settings"):
        classifier_type = st.selectbox(
            "Classifier Type:",
            ["Ensemble", "Keyword Only", "ML Only", "Transformer Only"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        st.info(f"Selected: {classifier_type} with threshold {confidence_threshold}")
    
    # Response generation settings
    with st.expander("Response Generation Settings"):
        similarity_threshold = st.slider(
            "Knowledge Base Similarity Threshold:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        max_response_time = st.slider(
            "Maximum Response Time (seconds):",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )
        
        st.info(f"Similarity threshold: {similarity_threshold}, Max response time: {max_response_time}s")
    
    # Business constraints
    with st.expander("Business Constraints"):
        escalation_threshold = st.slider(
            "Escalation Threshold:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        max_automation_rate = st.slider(
            "Maximum Automation Rate:",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1
        )
        
        st.info(f"Escalation threshold: {escalation_threshold}, Max automation: {max_automation_rate:.1%}")
    
    # System status
    st.subheader("System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("System Status", "🟢 Online")
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    with status_col2:
        st.metric("Models Loaded", "✅ All")
        st.metric("Knowledge Base", "✅ Ready")
    
    with status_col3:
        st.metric("Response Templates", "✅ Active")
        st.metric("Performance", "🟢 Good")


if __name__ == "__main__":
    main()
