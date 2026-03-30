"""
Evaluation metrics and performance tracking for customer support automation.

This module implements comprehensive evaluation metrics for measuring
the performance of the support automation system across multiple dimensions.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.schemas import (
    SupportTicket, IntentType, AutomatedResponse, ResponseType,
    IntentPrediction
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_response_time: float
    automation_rate: float
    escalation_rate: float
    customer_satisfaction: float
    resolution_rate: float
    confidence_score: float


class IntentClassificationEvaluator:
    """Evaluates intent classification performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.predictions: List[IntentPrediction] = []
        self.true_labels: List[IntentType] = []
        self.response_times: List[float] = []
        
    def add_prediction(self, prediction: IntentPrediction, true_label: IntentType, 
                      response_time: float):
        """Add a prediction for evaluation."""
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.response_times.append(response_time)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate classification metrics."""
        if not self.predictions:
            return {}
        
        # Extract predicted intents and confidences
        predicted_intents = [pred.intent for pred in self.predictions]
        confidences = [pred.confidence for pred in self.predictions]
        
        # Calculate basic metrics
        accuracy = accuracy_score(self.true_labels, predicted_intents)
        precision = precision_score(self.true_labels, predicted_intents, average='weighted', zero_division=0)
        recall = recall_score(self.true_labels, predicted_intents, average='weighted', zero_division=0)
        f1 = f1_score(self.true_labels, predicted_intents, average='weighted', zero_division=0)
        
        # Calculate confidence metrics
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Calculate response time metrics
        avg_response_time = np.mean(self.response_times)
        response_time_std = np.std(self.response_times)
        
        # Calculate per-class metrics
        class_report = classification_report(
            self.true_labels, predicted_intents, 
            output_dict=True, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'avg_response_time': avg_response_time,
            'response_time_std': response_time_std,
            'total_predictions': len(self.predictions),
            'class_report': class_report
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for intent classification."""
        if not self.predictions:
            return np.array([])
        
        predicted_intents = [pred.intent for pred in self.predictions]
        return confusion_matrix(self.true_labels, predicted_intents)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix()
        if cm.size == 0:
            logger.warning("No predictions to plot")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[intent.value for intent in IntentType],
            yticklabels=[intent.value for intent in IntentType]
        )
        plt.title('Intent Classification Confusion Matrix')
        plt.xlabel('Predicted Intent')
        plt.ylabel('True Intent')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


class ResponseGenerationEvaluator:
    """Evaluates response generation performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.responses: List[AutomatedResponse] = []
        self.feedback_data: List[Dict[str, Any]] = []
        
    def add_response(self, response: AutomatedResponse):
        """Add a response for evaluation."""
        self.responses.append(response)
    
    def add_feedback(self, response_id: str, satisfaction_rating: int, 
                    resolved: bool, escalation_needed: bool):
        """Add feedback for a response."""
        feedback = {
            'response_id': response_id,
            'satisfaction_rating': satisfaction_rating,
            'resolved': resolved,
            'escalation_needed': escalation_needed,
            'timestamp': datetime.now()
        }
        self.feedback_data.append(feedback)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate response generation metrics."""
        if not self.responses:
            return {}
        
        # Basic response metrics
        total_responses = len(self.responses)
        automated_responses = sum(1 for r in self.responses if r.response_type == ResponseType.AUTOMATED)
        escalated_responses = sum(1 for r in self.responses if r.response_type == ResponseType.ESCALATED)
        
        automation_rate = automated_responses / total_responses if total_responses > 0 else 0
        escalation_rate = escalated_responses / total_responses if total_responses > 0 else 0
        
        # Response time metrics
        response_times = [r.processing_time for r in self.responses]
        avg_response_time = np.mean(response_times)
        
        # Confidence metrics
        confidences = [r.confidence for r in self.responses]
        avg_confidence = np.mean(confidences)
        
        # Feedback metrics (if available)
        feedback_metrics = {}
        if self.feedback_data:
            feedback_df = pd.DataFrame(self.feedback_data)
            feedback_metrics = {
                'avg_satisfaction': feedback_df['satisfaction_rating'].mean(),
                'resolution_rate': feedback_df['resolved'].mean(),
                'escalation_rate_feedback': feedback_df['escalation_needed'].mean(),
                'total_feedback': len(feedback_df)
            }
        
        metrics = {
            'total_responses': total_responses,
            'automation_rate': automation_rate,
            'escalation_rate': escalation_rate,
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            **feedback_metrics
        }
        
        return metrics


class BusinessMetricsEvaluator:
    """Evaluates business impact metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.tickets: List[SupportTicket] = []
        self.responses: List[AutomatedResponse] = []
        self.costs: Dict[str, float] = {
            'human_agent_cost_per_hour': 25.0,
            'automated_response_cost': 0.10,
            'escalation_cost': 5.0
        }
        
    def add_ticket(self, ticket: SupportTicket):
        """Add a ticket for evaluation."""
        self.tickets.append(ticket)
    
    def add_response(self, response: AutomatedResponse):
        """Add a response for evaluation."""
        self.responses.append(response)
    
    def calculate_cost_savings(self) -> Dict[str, float]:
        """Calculate cost savings from automation."""
        if not self.responses:
            return {}
        
        total_responses = len(self.responses)
        automated_responses = sum(1 for r in self.responses if r.response_type == ResponseType.AUTOMATED)
        escalated_responses = sum(1 for r in self.responses if r.response_type == ResponseType.ESCALATED)
        
        # Calculate costs
        automated_cost = automated_responses * self.costs['automated_response_cost']
        escalation_cost = escalated_responses * self.costs['escalation_cost']
        human_cost = total_responses * (self.costs['human_agent_cost_per_hour'] / 60)  # Assuming 1 minute per response
        
        total_cost_with_automation = automated_cost + escalation_cost
        total_cost_without_automation = human_cost
        
        cost_savings = total_cost_without_automation - total_cost_with_automation
        cost_savings_percentage = (cost_savings / total_cost_without_automation) * 100 if total_cost_without_automation > 0 else 0
        
        return {
            'total_responses': total_responses,
            'automated_responses': automated_responses,
            'escalated_responses': escalated_responses,
            'automated_cost': automated_cost,
            'escalation_cost': escalation_cost,
            'human_cost': human_cost,
            'total_cost_with_automation': total_cost_with_automation,
            'total_cost_without_automation': total_cost_without_automation,
            'cost_savings': cost_savings,
            'cost_savings_percentage': cost_savings_percentage
        }
    
    def calculate_service_level_metrics(self) -> Dict[str, float]:
        """Calculate service level metrics."""
        if not self.tickets:
            return {}
        
        # Resolution time metrics
        resolved_tickets = [t for t in self.tickets if t.resolution_time is not None]
        if resolved_tickets:
            resolution_times = [t.resolution_time for t in resolved_tickets]
            avg_resolution_time = np.mean(resolution_times)
            median_resolution_time = np.median(resolution_times)
            resolution_time_std = np.std(resolution_times)
        else:
            avg_resolution_time = 0
            median_resolution_time = 0
            resolution_time_std = 0
        
        # Customer satisfaction metrics
        satisfaction_ratings = [t.customer_satisfaction for t in self.tickets if t.customer_satisfaction is not None]
        if satisfaction_ratings:
            avg_satisfaction = np.mean(satisfaction_ratings)
            satisfaction_std = np.std(satisfaction_ratings)
        else:
            avg_satisfaction = 0
            satisfaction_std = 0
        
        # Ticket volume metrics
        total_tickets = len(self.tickets)
        open_tickets = sum(1 for t in self.tickets if t.status.value == 'open')
        resolved_tickets_count = len(resolved_tickets)
        
        return {
            'total_tickets': total_tickets,
            'open_tickets': open_tickets,
            'resolved_tickets': resolved_tickets_count,
            'resolution_rate': resolved_tickets_count / total_tickets if total_tickets > 0 else 0,
            'avg_resolution_time': avg_resolution_time,
            'median_resolution_time': median_resolution_time,
            'resolution_time_std': resolution_time_std,
            'avg_customer_satisfaction': avg_satisfaction,
            'satisfaction_std': satisfaction_std
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluation system that combines all metrics."""
    
    def __init__(self):
        """Initialize the comprehensive evaluator."""
        self.intent_evaluator = IntentClassificationEvaluator()
        self.response_evaluator = ResponseGenerationEvaluator()
        self.business_evaluator = BusinessMetricsEvaluator()
        
    def evaluate_system(
        self, 
        tickets: List[SupportTicket], 
        responses: List[AutomatedResponse],
        feedback_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate the entire system comprehensively."""
        
        # Add data to evaluators
        for ticket in tickets:
            self.business_evaluator.add_ticket(ticket)
        
        for response in responses:
            self.response_evaluator.add_response(response)
            self.business_evaluator.add_response(response)
        
        # Add feedback data
        if feedback_data:
            for feedback in feedback_data:
                self.response_evaluator.add_feedback(**feedback)
        
        # Calculate all metrics
        intent_metrics = self.intent_evaluator.calculate_metrics()
        response_metrics = self.response_evaluator.calculate_metrics()
        cost_metrics = self.business_evaluator.calculate_cost_savings()
        service_metrics = self.business_evaluator.calculate_service_level_metrics()
        
        # Combine metrics
        comprehensive_metrics = {
            'intent_classification': intent_metrics,
            'response_generation': response_metrics,
            'cost_analysis': cost_metrics,
            'service_level': service_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_metrics
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("CUSTOMER SUPPORT AUTOMATION EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {metrics['evaluation_timestamp']}")
        report.append("")
        
        # Intent Classification Metrics
        intent_metrics = metrics.get('intent_classification', {})
        if intent_metrics:
            report.append("INTENT CLASSIFICATION METRICS")
            report.append("-" * 30)
            report.append(f"Accuracy: {intent_metrics.get('accuracy', 0):.3f}")
            report.append(f"Precision: {intent_metrics.get('precision', 0):.3f}")
            report.append(f"Recall: {intent_metrics.get('recall', 0):.3f}")
            report.append(f"F1-Score: {intent_metrics.get('f1_score', 0):.3f}")
            report.append(f"Average Confidence: {intent_metrics.get('avg_confidence', 0):.3f}")
            report.append(f"Average Response Time: {intent_metrics.get('avg_response_time', 0):.3f}s")
            report.append("")
        
        # Response Generation Metrics
        response_metrics = metrics.get('response_generation', {})
        if response_metrics:
            report.append("RESPONSE GENERATION METRICS")
            report.append("-" * 30)
            report.append(f"Total Responses: {response_metrics.get('total_responses', 0)}")
            report.append(f"Automation Rate: {response_metrics.get('automation_rate', 0):.1%}")
            report.append(f"Escalation Rate: {response_metrics.get('escalation_rate', 0):.1%}")
            report.append(f"Average Response Time: {response_metrics.get('avg_response_time', 0):.3f}s")
            report.append(f"Average Confidence: {response_metrics.get('avg_confidence', 0):.3f}")
            
            if 'avg_satisfaction' in response_metrics:
                report.append(f"Average Satisfaction: {response_metrics['avg_satisfaction']:.1f}/5")
                report.append(f"Resolution Rate: {response_metrics.get('resolution_rate', 0):.1%}")
            report.append("")
        
        # Cost Analysis
        cost_metrics = metrics.get('cost_analysis', {})
        if cost_metrics:
            report.append("COST ANALYSIS")
            report.append("-" * 30)
            report.append(f"Total Responses: {cost_metrics.get('total_responses', 0)}")
            report.append(f"Automated Responses: {cost_metrics.get('automated_responses', 0)}")
            report.append(f"Escalated Responses: {cost_metrics.get('escalated_responses', 0)}")
            report.append(f"Cost Savings: ${cost_metrics.get('cost_savings', 0):.2f}")
            report.append(f"Cost Savings Percentage: {cost_metrics.get('cost_savings_percentage', 0):.1f}%")
            report.append("")
        
        # Service Level Metrics
        service_metrics = metrics.get('service_level', {})
        if service_metrics:
            report.append("SERVICE LEVEL METRICS")
            report.append("-" * 30)
            report.append(f"Total Tickets: {service_metrics.get('total_tickets', 0)}")
            report.append(f"Resolution Rate: {service_metrics.get('resolution_rate', 0):.1%}")
            report.append(f"Average Resolution Time: {service_metrics.get('avg_resolution_time', 0):.1f} minutes")
            report.append(f"Average Customer Satisfaction: {service_metrics.get('avg_customer_satisfaction', 0):.1f}/5")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_metrics(self, metrics: Dict[str, Any], filepath: str):
        """Save metrics to a JSON file."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert all numpy types
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(item) for item in d]
            else:
                return convert_numpy(d)
        
        converted_metrics = recursive_convert(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(converted_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


class PerformanceTracker:
    """Tracks system performance over time."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.metrics_history: List[Dict[str, Any]] = []
        
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time."""
        if not self.metrics_history:
            return {}
        
        trends = {}
        
        # Extract key metrics over time
        for metric_name in ['accuracy', 'automation_rate', 'avg_satisfaction', 'cost_savings']:
            values = []
            for metrics in self.metrics_history:
                # Navigate nested structure to find metric
                value = self._extract_nested_metric(metrics, metric_name)
                if value is not None:
                    values.append(value)
            
            if values:
                trends[metric_name] = values
        
        return trends
    
    def _extract_nested_metric(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract a metric from nested structure."""
        for section in ['intent_classification', 'response_generation', 'cost_analysis', 'service_level']:
            if section in metrics and metric_name in metrics[section]:
                return metrics[section][metric_name]
        return None
    
    def plot_performance_trends(self, save_path: Optional[str] = None):
        """Plot performance trends over time."""
        trends = self.get_performance_trends()
        
        if not trends:
            logger.warning("No performance trends to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(trends.items()):
            if i < 4:  # Limit to 4 subplots
                axes[i].plot(values, marker='o')
                axes[i].set_title(f'{metric_name.replace("_", " ").title()} Over Time')
                axes[i].set_xlabel('Time Period')
                axes[i].set_ylabel(metric_name.replace("_", " ").title())
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
