# Customer Support Automation System

A research-focused customer support automation system that demonstrates advanced NLP and ML techniques for intent classification, response generation, and knowledge base management.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational demonstration system.**

- This system is for educational purposes only
- Do not use for automated decision-making without human review
- All responses should be validated by human agents
- This system is not production-ready
- Use only for learning and research purposes

## Overview

This project implements a comprehensive customer support automation system that includes:

- **Intent Classification**: Multiple approaches from keyword matching to transformer-based models
- **Response Generation**: Template-based and knowledge base-driven response generation
- **Knowledge Base Management**: Semantic search and similarity matching
- **Performance Evaluation**: Comprehensive metrics and business impact analysis
- **Interactive Demo**: Streamlit-based web interface for testing and demonstration

## Features

### Intent Classification
- Keyword-based classification
- Machine learning models (Logistic Regression, Random Forest)
- Transformer-based models (DistilBERT)
- Ensemble methods combining multiple approaches

### Response Generation
- Template-based responses with variable substitution
- Knowledge base retrieval and similarity matching
- Confidence-based escalation logic
- Response optimization based on feedback

### Knowledge Base Management
- Semantic search using sentence transformers
- Category-based organization
- Usage tracking and success rate monitoring
- Dynamic content updates

### Evaluation & Metrics
- Intent classification accuracy, precision, recall, F1-score
- Response generation metrics (automation rate, escalation rate)
- Business impact analysis (cost savings, service levels)
- Customer satisfaction tracking

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Customer-Support-Automation-System.git
cd Customer-Support-Automation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Optional: Development Setup

For development with additional tools:
```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

### 1. Generate Synthetic Data
```bash
python main.py generate-data --n-tickets 1000
```

### 2. Train Models
```bash
python main.py train-models
```

### 3. Evaluate System
```bash
python main.py evaluate
```

### 4. Run Complete Pipeline
```bash
python main.py pipeline
```

### 5. Launch Interactive Demo
```bash
python main.py demo
```

The demo will be available at `http://localhost:8501`

## Usage

### Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
# Generate synthetic data
python main.py generate-data --n-tickets 1000 --data-dir data/synthetic

# Train models
python main.py train-models --data-dir data/synthetic --model-dir models

# Evaluate system performance
python main.py evaluate --data-dir data/synthetic --model-dir models

# Run complete pipeline
python main.py pipeline --n-tickets 1000

# Launch interactive demo
python main.py demo
```

### Programmatic Usage

```python
from src.data.generator import SyntheticDataGenerator
from src.models.intent_classifier import IntentClassifierEnsemble
from src.models.response_generator import KnowledgeBaseManager, SupportAutomationSystem
from src.data.schemas import create_synthetic_ticket

# Generate synthetic data
generator = SyntheticDataGenerator()
tickets = generator.generate_tickets(100)
kb_entries = generator.generate_knowledge_base()

# Initialize system
kb_manager = KnowledgeBaseManager()
kb_manager.add_entries(kb_entries)

intent_classifier = IntentClassifierEnsemble()
intent_classifier.train(tickets)

system = SupportAutomationSystem(intent_classifier, kb_manager)

# Process a ticket
ticket = create_synthetic_ticket(
    ticket_id="TICKET_001",
    customer_id="CUSTOMER_001",
    subject="Order Status Inquiry",
    message="What's the status of my order?"
)

response = system.process_ticket(ticket)
print(f"Intent: {response.intent.value}")
print(f"Response: {response.response_text}")
print(f"Confidence: {response.confidence:.3f}")
```

## Data Schema

### Support Tickets
- `ticket_id`: Unique identifier
- `customer_id`: Customer identifier
- `timestamp`: Creation timestamp
- `subject`: Ticket subject
- `message`: Customer message
- `priority`: Priority level (low, medium, high, urgent)
- `status`: Ticket status (open, in_progress, resolved, closed, escalated)
- `category`: Intent category
- `resolution_time`: Time to resolution (minutes)
- `customer_satisfaction`: Satisfaction rating (1-5)
- `tags`: Additional tags
- `metadata`: Additional metadata

### Knowledge Base Entries
- `entry_id`: Unique identifier
- `title`: Entry title
- `content`: Entry content
- `category`: Entry category
- `tags`: Associated tags
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `usage_count`: Number of times used
- `success_rate`: Success rate (0-1)
- `metadata`: Additional metadata

## Configuration

The system uses YAML configuration files located in `configs/`:

```yaml
# configs/config.yaml
models:
  intent_classifier:
    model_name: "distilbert-base-uncased"
    max_length: 128
    batch_size: 32
    learning_rate: 2e-5
    num_epochs: 3

evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "response_time"
    - "customer_satisfaction"
  
  test_split: 0.2
  validation_split: 0.2
  random_seed: 42

constraints:
  max_response_time: 5.0
  min_confidence_threshold: 0.7
  escalation_threshold: 0.5
  max_automation_rate: 0.8
```

## Evaluation Metrics

### Intent Classification
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for each intent class
- **Recall**: Recall for each intent class
- **F1-Score**: F1-score for each intent class
- **Confidence**: Average confidence scores

### Response Generation
- **Automation Rate**: Percentage of automated responses
- **Escalation Rate**: Percentage of escalated tickets
- **Response Time**: Average processing time
- **Customer Satisfaction**: Average satisfaction ratings

### Business Impact
- **Cost Savings**: Financial impact of automation
- **Service Level**: Resolution rates and times
- **Agent Utilization**: Efficiency improvements
- **Customer Experience**: Satisfaction metrics

## Project Structure

```
customer-support-automation/
├── src/
│   ├── data/
│   │   ├── schemas.py          # Data structures and schemas
│   │   └── generator.py        # Synthetic data generation
│   ├── models/
│   │   ├── intent_classifier.py    # Intent classification models
│   │   └── response_generator.py   # Response generation system
│   ├── eval/
│   │   └── metrics.py          # Evaluation metrics and analysis
│   ├── viz/
│   │   └── demo.py            # Streamlit demo application
│   └── utils/                 # Utility functions
├── configs/
│   └── config.yaml           # Configuration files
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data files
│   └── synthetic/            # Generated synthetic data
├── models/                   # Trained model files
├── results/                  # Evaluation results
├── logs/                     # Log files
├── tests/                    # Test files
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Utility scripts
├── assets/                   # Static assets
├── demo/                     # Demo files
├── main.py                   # Main CLI script
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Development

### Code Quality
The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and code analysis
- **Pre-commit**: Git hooks for quality checks

Run formatting:
```bash
black src/ tests/
ruff check src/ tests/
```

### Adding New Features

1. **New Intent Types**: Add to `IntentType` enum in `src/data/schemas.py`
2. **New Models**: Implement in `src/models/intent_classifier.py`
3. **New Metrics**: Add to `src/eval/metrics.py`
4. **New Templates**: Add to `src/models/response_generator.py`

## Limitations

- **Synthetic Data**: Uses generated data for demonstration
- **Model Performance**: May not generalize to real-world data
- **Scalability**: Not optimized for high-volume production use
- **Security**: No authentication or authorization mechanisms
- **Compliance**: No built-in privacy or compliance features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{customer_support_automation,
  title={Customer Support Automation System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Customer-Support-Automation-System}
}
```

## Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation
- Review the demo application

## Acknowledgments

- Hugging Face Transformers library
- Streamlit for the demo interface
- Scikit-learn for ML models
- Sentence Transformers for semantic search
- The open-source community for various libraries and tools
# Customer-Support-Automation-System
