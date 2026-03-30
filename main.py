#!/usr/bin/env python3
"""
Main training and evaluation script for customer support automation system.

This script provides a command-line interface for training models, generating
synthetic data, and evaluating system performance.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.generator import SyntheticDataGenerator
from data.schemas import tickets_to_dataframe, dataframe_to_tickets
from models.intent_classifier import (
    KeywordIntentClassifier, MLIntentClassifier, 
    TransformerIntentClassifier, IntentClassifierEnsemble
)
from models.response_generator import (
    KnowledgeBaseManager, ResponseGenerator, SupportAutomationSystem
)
from eval.metrics import ComprehensiveEvaluator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/support_automation.log')
        ]
    )


def generate_data(output_dir: str = "data/synthetic", n_tickets: int = 1000) -> None:
    """Generate synthetic data for training and testing."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {n_tickets} synthetic tickets...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    generator = SyntheticDataGenerator(seed=42)
    tickets_df, kb_df, interactions_df = generator.save_synthetic_data(output_dir)
    
    logger.info(f"Generated data saved to {output_dir}/")
    logger.info(f"- {len(tickets_df)} support tickets")
    logger.info(f"- {len(kb_df)} knowledge base entries")
    logger.info(f"- {len(interactions_df)} customer interactions")


def train_models(data_dir: str = "data/synthetic", model_dir: str = "models") -> Dict[str, Any]:
    """Train all models in the system."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    tickets_df = pd.read_csv(f"{data_dir}/support_tickets.csv")
    kb_df = pd.read_csv(f"{data_dir}/knowledge_base.csv")
    
    # Convert to objects
    tickets = dataframe_to_tickets(tickets_df)
    
    # Initialize knowledge base
    kb_manager = KnowledgeBaseManager()
    kb_entries = []
    for _, row in kb_df.iterrows():
        from data.schemas import KnowledgeBaseEntry
        entry = KnowledgeBaseEntry(
            entry_id=row['entry_id'],
            title=row['title'],
            content=row['content'],
            category=row['category'],
            tags=row['tags'].split(',') if pd.notna(row['tags']) else [],
            created_at=pd.to_datetime(row['created_at']),
            updated_at=pd.to_datetime(row['updated_at']),
            usage_count=row['usage_count'],
            success_rate=row['success_rate'],
            metadata=eval(row['metadata']) if pd.notna(row['metadata']) else {}
        )
        kb_entries.append(entry)
    
    kb_manager.add_entries(kb_entries)
    
    # Train intent classifiers
    results = {}
    
    # Keyword classifier (no training needed)
    keyword_classifier = KeywordIntentClassifier()
    results['keyword'] = {'status': 'ready', 'accuracy': 0.0}
    
    # ML classifier
    try:
        ml_classifier = MLIntentClassifier()
        ml_results = ml_classifier.train(tickets)
        ml_classifier.save_model(f"{model_dir}/ml_classifier.pkl")
        results['ml'] = ml_results
        logger.info(f"ML classifier trained with accuracy: {ml_results['accuracy']:.3f}")
    except Exception as e:
        logger.error(f"ML classifier training failed: {e}")
        results['ml'] = {'status': 'failed', 'error': str(e)}
    
    # Transformer classifier
    try:
        transformer_classifier = TransformerIntentClassifier()
        transformer_results = transformer_classifier.train(tickets)
        transformer_classifier.save_model(f"{model_dir}/transformer_classifier")
        results['transformer'] = transformer_results
        logger.info(f"Transformer classifier trained")
    except Exception as e:
        logger.error(f"Transformer classifier training failed: {e}")
        results['transformer'] = {'status': 'failed', 'error': str(e)}
    
    # Ensemble classifier
    try:
        ensemble_classifier = IntentClassifierEnsemble()
        ensemble_results = ensemble_classifier.train(tickets)
        results['ensemble'] = ensemble_results
        logger.info("Ensemble classifier trained")
    except Exception as e:
        logger.error(f"Ensemble classifier training failed: {e}")
        results['ensemble'] = {'status': 'failed', 'error': str(e)}
    
    # Initialize response generator
    response_generator = ResponseGenerator(kb_manager)
    
    # Create automation system
    system = SupportAutomationSystem(ensemble_classifier, kb_manager)
    
    # Save system components
    import pickle
    with open(f"{model_dir}/kb_manager.pkl", 'wb') as f:
        pickle.dump(kb_manager, f)
    
    with open(f"{model_dir}/response_generator.pkl", 'wb') as f:
        pickle.dump(response_generator, f)
    
    logger.info("Model training completed")
    return results


def evaluate_system(data_dir: str = "data/synthetic", model_dir: str = "models") -> Dict[str, Any]:
    """Evaluate the trained system."""
    logger = logging.getLogger(__name__)
    logger.info("Starting system evaluation...")
    
    # Load data
    tickets_df = pd.read_csv(f"{data_dir}/support_tickets.csv")
    tickets = dataframe_to_tickets(tickets_df)
    
    # Load models
    import pickle
    
    # Load knowledge base manager
    with open(f"{model_dir}/kb_manager.pkl", 'rb') as f:
        kb_manager = pickle.load(f)
    
    # Load response generator
    with open(f"{model_dir}/response_generator.pkl", 'rb') as f:
        response_generator = pickle.load(f)
    
    # Initialize system (use ensemble classifier)
    ensemble_classifier = IntentClassifierEnsemble()
    system = SupportAutomationSystem(ensemble_classifier, kb_manager)
    
    # Generate responses for evaluation
    logger.info("Generating responses for evaluation...")
    responses = []
    evaluation_tickets = tickets[:100]  # Use first 100 tickets for evaluation
    
    for i, ticket in enumerate(evaluation_tickets):
        try:
            response = system.process_ticket(ticket)
            responses.append(response)
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(evaluation_tickets)} tickets")
        except Exception as e:
            logger.warning(f"Failed to process ticket {ticket.ticket_id}: {e}")
    
    # Evaluate system
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate_system(evaluation_tickets, responses)
    
    # Generate report
    report = evaluator.generate_report(metrics)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    evaluator.save_metrics(metrics, "results/evaluation_metrics.json")
    
    with open("results/evaluation_report.txt", 'w') as f:
        f.write(report)
    
    logger.info("Evaluation completed")
    logger.info(f"Results saved to results/")
    
    return metrics


def run_demo() -> None:
    """Run the Streamlit demo application."""
    import subprocess
    import sys
    
    demo_path = Path(__file__).parent / "src" / "viz" / "demo.py"
    
    if not demo_path.exists():
        logger.error(f"Demo file not found: {demo_path}")
        return
    
    logger.info("Starting Streamlit demo...")
    logger.info("Open your browser to http://localhost:8501")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start demo: {e}")
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Customer Support Automation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic data
  python main.py generate-data --n-tickets 1000
  
  # Train all models
  python main.py train-models
  
  # Evaluate system
  python main.py evaluate
  
  # Run complete pipeline
  python main.py pipeline
  
  # Run demo
  python main.py demo
        """
    )
    
    parser.add_argument(
        "command",
        choices=["generate-data", "train-models", "evaluate", "pipeline", "demo"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--data-dir",
        default="data/synthetic",
        help="Directory for data files (default: data/synthetic)"
    )
    
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory for model files (default: models)"
    )
    
    parser.add_argument(
        "--n-tickets",
        type=int,
        default=1000,
        help="Number of synthetic tickets to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Customer Support Automation System")
    logger.info("=" * 50)
    
    try:
        if args.command == "generate-data":
            generate_data(args.data_dir, args.n_tickets)
            
        elif args.command == "train-models":
            train_models(args.data_dir, args.model_dir)
            
        elif args.command == "evaluate":
            evaluate_system(args.data_dir, args.model_dir)
            
        elif args.command == "pipeline":
            logger.info("Running complete pipeline...")
            generate_data(args.data_dir, args.n_tickets)
            train_models(args.data_dir, args.model_dir)
            evaluate_system(args.data_dir, args.model_dir)
            logger.info("Pipeline completed successfully!")
            
        elif args.command == "demo":
            run_demo()
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
