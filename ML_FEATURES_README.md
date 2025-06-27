# AI News Dashboard - ML Features Documentation

## Overview

This document describes the advanced Machine Learning features implemented in the AI News Dashboard, including transformer-based classification and ML prediction services.

## 🤖 Transformer-Based News Classification

### Features
- **BERT/DistilBERT-based text classification** for accurate topic categorization
- **Multi-label classification** supporting multiple categories per article
- **Sentiment analysis integration** using RoBERTa models
- **Named Entity Recognition (NER)** for extracting key entities
- **Confidence scoring** and uncertainty estimation
- **Batch processing** for efficient handling of multiple articles
- **Redis caching** for improved performance
- **Prometheus metrics** for monitoring

### Implementation

#### Files
- `news/transformer_classifier.py` - Main transformer classification service
- `news/ingest.js` - Updated news ingestion with transformer integration

#### Usage

```python
from news.transformer_classifier import classify_article

# Classify a single article
result = await classify_article(
    title="Apple Unveils New iPhone with AI Features",
    description="Latest iPhone model includes cutting-edge AI",
    content="Apple announced new iPhone features...",
    source="TechNews"
)

print(f"Category: {result.primary_category}")
print(f"Confidence: {result.confidence_score}")
print(f"Sentiment: {result.sentiment}")
```

#### Categories Supported
- Technology
- Business
- Politics
- Health
- Science
- Sports
- Entertainment
- World
- Environment
- Education
- Crime
- Weather
- Travel
- Food
- Lifestyle

### Configuration

The transformer classifier can be configured via the config dictionary:

```python
config = {
    'classification_model': 'distilbert-base-uncased',
    'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'ner_model': 'en_core_web_sm',
    'use_gpu': True,
    'batch_size': 16,
    'max_length': 512,
    'cache_ttl': 3600,
    'confidence_threshold': 0.7
}
```

## 📈 ML Prediction Service

### Features
- **Trend Forecasting** using LSTM and Transformer models
- **Article Popularity Prediction** with Random Forest and Neural Networks
- **Sentiment Trend Analysis** for market sentiment tracking
- **Topic Emergence Detection** for identifying new trending topics
- **User Engagement Prediction** for content optimization
- **Real-time and Batch APIs** for different use cases
- **Model Versioning** and A/B testing support
- **Performance Monitoring** with detailed metrics

### Implementation

#### Files
- `mlops/ml_prediction_service.py` - Core ML prediction service
- `mlops/ml_prediction_cli.py` - Command-line interface for scheduled tasks
- `scheduler/jobs.yaml` - Scheduled ML prediction jobs

#### Prediction Types

1. **Trend Forecasting**
   - Predicts future news trends based on historical data
   - Uses LSTM networks for time series analysis
   - Supports multiple time horizons (hours, days, weeks)

2. **Popularity Prediction**
   - Estimates article engagement potential
   - Considers title, content, source, and timing
   - Uses ensemble methods for robust predictions

3. **Sentiment Trends**
   - Analyzes sentiment patterns over time
   - Identifies sentiment shifts and anomalies
   - Provides market sentiment indicators

4. **Topic Emergence**
   - Detects new and emerging topics
   - Uses clustering and novelty detection
   - Alerts for breaking news categories

5. **User Engagement**
   - Predicts user interaction patterns
   - Optimizes content recommendation
   - Supports personalization features

### Usage

#### Python API

```python
from mlops.ml_prediction_service import MLPredictionService, PredictionRequest, PredictionType

# Initialize service
service = MLPredictionService()
await service.initialize()

# Create prediction request
request = PredictionRequest(
    prediction_type=PredictionType.TREND_FORECASTING,
    time_horizon=24,  # 24 hours
    include_confidence=True
)

# Get prediction
result = await service.predict(request)
print(f"Predicted trends: {result.predictions['trends']}")
```

#### Command Line Interface

```bash
# Trend forecasting
python mlops/ml_prediction_cli.py --task=trend_forecasting --horizon=24h

# Popularity prediction
python mlops/ml_prediction_cli.py --task=popularity_prediction

# Sentiment trends
python mlops/ml_prediction_cli.py --task=sentiment_trends

# Topic emergence
python mlops/ml_prediction_cli.py --task=topic_emergence
```

### Scheduled Jobs

ML prediction tasks are automatically scheduled via the job scheduler:

- **Trend Forecasting**: Every 6 hours
- **Popularity Prediction**: Every 4 hours
- **Sentiment Trends**: Every 2 hours
- **Topic Emergence**: Twice daily (8 AM, 8 PM)

## 🔧 Installation & Setup

### Prerequisites

1. **Python 3.8+** with pip
2. **Redis** for caching
3. **PostgreSQL** for data storage
4. **CUDA** (optional, for GPU acceleration)

### Installation Steps

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Set up environment variables**:
   ```bash
   export REDIS_HOST=localhost
   export REDIS_PORT=6379
   export POSTGRES_URL=postgresql://user:pass@localhost/newsdb
   ```

4. **Initialize ML models**:
   ```bash
   python mlops/ml_prediction_cli.py --task=trend_forecasting --horizon=1h
   ```

### Configuration

Create a `config.json` file for ML services:

```json
{
  "classification_model": "distilbert-base-uncased",
  "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "ner_model": "en_core_web_sm",
  "use_gpu": true,
  "batch_size": 16,
  "max_length": 512,
  "cache_ttl": 3600,
  "confidence_threshold": 0.7,
  "redis_host": "localhost",
  "redis_port": 6379,
  "redis_db": 3
}
```

## 📊 Monitoring & Metrics

### Prometheus Metrics

The ML services expose the following metrics:

- `transformer_classifications_total` - Total classifications made
- `transformer_classification_duration_seconds` - Classification latency
- `transformer_confidence_scores` - Classification confidence distribution
- `ml_predictions_total` - Total ML predictions made
- `ml_prediction_duration_seconds` - Prediction latency
- `ml_model_accuracy` - Current model accuracy

### Grafana Dashboards

Pre-configured dashboards are available for:
- Classification performance
- Prediction accuracy trends
- System resource usage
- Error rates and alerts

## 🧪 Testing

### Unit Tests

```bash
# Run transformer classifier tests
pytest tests/test_transformer_classifier.py

# Run ML prediction service tests
pytest tests/test_ml_prediction_service.py

# Run integration tests
pytest tests/test_ml_integration.py
```

### Performance Testing

```bash
# Benchmark classification performance
python tests/benchmark_classification.py

# Load test prediction service
python tests/load_test_predictions.py
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Use the provided Dockerfile
docker build -t ai-news-ml .
docker run -p 8000:8000 ai-news-ml
```

### Kubernetes Deployment

```yaml
# Apply Kubernetes manifests
kubectl apply -f k8s/ml-services.yaml
kubectl apply -f k8s/transformer-classifier.yaml
```

### Production Considerations

1. **GPU Resources**: Ensure CUDA-compatible GPUs for optimal performance
2. **Memory Requirements**: Minimum 8GB RAM for transformer models
3. **Redis Clustering**: Use Redis cluster for high availability
4. **Model Caching**: Implement model caching for faster startup
5. **Load Balancing**: Use multiple instances for high throughput

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in configuration
   - Use CPU-only mode: `"use_gpu": false`

2. **Model Download Failures**:
   - Check internet connectivity
   - Verify Hugging Face model names
   - Use local model cache

3. **Redis Connection Issues**:
   - Verify Redis server is running
   - Check connection parameters
   - Ensure Redis version compatibility

4. **Classification Accuracy Issues**:
   - Review training data quality
   - Adjust confidence thresholds
   - Consider model fine-tuning

### Logs and Debugging

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# View classification logs
tail -f logs/transformer_classifier.log

# View prediction service logs
tail -f logs/ml_prediction_service.log
```

## 📚 API Reference

### Transformer Classifier API

#### `classify_article(title, description, content, source, url)`
Classifies a single news article.

**Parameters:**
- `title` (str): Article title
- `description` (str): Article description
- `content` (str): Article content
- `source` (str): News source
- `url` (str): Article URL

**Returns:**
- `ClassificationResult`: Classification results with confidence scores

#### `classify_articles_batch(articles)`
Classifies multiple articles in batch.

**Parameters:**
- `articles` (List[Dict]): List of article dictionaries

**Returns:**
- `List[ClassificationResult]`: List of classification results

### ML Prediction Service API

#### `predict(request)`
Makes ML predictions based on request type.

**Parameters:**
- `request` (PredictionRequest): Prediction request object

**Returns:**
- `PredictionResult`: Prediction results with metadata

#### `batch_predict(requests)`
Makes multiple predictions in batch.

**Parameters:**
- `requests` (List[PredictionRequest]): List of prediction requests

**Returns:**
- `List[PredictionResult]`: List of prediction results

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run tests: `pytest`
6. Submit a pull request

### Code Style

- Use Black for code formatting: `black .`
- Follow PEP 8 guidelines
- Add type hints for all functions
- Include docstrings for public methods

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation
- Contact the development team

---

**Last Updated**: December 2024
**Version**: 1.0.0