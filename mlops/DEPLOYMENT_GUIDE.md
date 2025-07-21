# MLOps Deployment Guide

## Overview

Quick deployment guide for the MLOps infrastructure.

## Prerequisites

- Python 3.8+
- Docker (optional)
- Git

## Local Deployment

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd ai-news-dashboard/mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy config template
cp agentic_config.yaml.example agentic_config.yaml

# Edit configuration
# Update settings as needed
```

### 3. Run Services

```bash
# Start code optimizer
python agentic_code_optimizer.py

# Run CI/CD pipeline
python headless_pipeline.py

# Generate documentation
python agentic_docs_generator.py
```

## Docker Deployment

### Build Image

```bash
docker build -t mlops-dashboard .
```

### Run Container

```bash
docker run -d -p 8080:8080 mlops-dashboard
```

## Production Deployment

### Environment Variables

```bash
export MLOPS_ENV=production
export LOG_LEVEL=info
export WORKERS=4
```

### Health Checks

- Service status: `/health`
- Metrics: `/metrics`
- Ready state: `/ready`

## Monitoring

- Logs: Check application logs for errors
- Metrics: Monitor system performance
- Alerts: Configure notifications for failures

## Troubleshooting

### Common Issues

1. **Import errors**: Check Python dependencies
2. **Config errors**: Validate YAML syntax
3. **Permission errors**: Check file permissions

### Debug Mode

```bash
python agentic_code_optimizer.py --debug --verbose
```

## Support

Refer to the main project documentation for detailed setup instructions.