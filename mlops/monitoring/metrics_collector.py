#!/usr/bin/env python3
"""
Dr. NewsForge's Advanced MLOps Monitoring & Observability System

Features:
- Real-time model performance monitoring
- Data drift detection with statistical tests
- Prometheus metrics collection
- Custom business metrics tracking
- Automated alerting and anomaly detection
- Performance profiling and optimization insights

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest,
    start_http_server
)
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import psutil
import GPUtil
import redis
import mlflow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Structure for model prediction data."""
    model_name: str
    prediction: Any
    confidence: float
    latency_ms: float
    input_features: Dict
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class ModelMetrics:
    """Structure for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float

class PrometheusMetrics:
    """Prometheus metrics collector for ML models."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Model performance metrics
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of predictions made',
            ['model_name', 'status'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_drift_score = Gauge(
            'ml_model_drift_score',
            'Data drift score for model',
            ['model_name', 'feature'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'system_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'system_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id'],
            registry=self.registry
        )
        
        # Business metrics
        self.news_articles_processed = Counter(
            'news_articles_processed_total',
            'Total number of news articles processed',
            ['source', 'category'],
            registry=self.registry
        )
        
        self.summarization_quality = Histogram(
            'summarization_quality_score',
            'Quality score of generated summaries',
            ['model_version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.user_engagement = Counter(
            'user_engagement_total',
            'User engagement events',
            ['event_type', 'feature'],
            registry=self.registry
        )

class DriftDetector:
    """Advanced data drift detection system."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        self.reference_data = reference_data
        self.threshold = threshold
        self.feature_stats = self._compute_reference_stats()
        
    def _compute_reference_stats(self) -> Dict[str, Dict]:
        """Compute reference statistics for drift detection."""
        stats = {}
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['int64', 'float64']:
                stats[column] = {
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'min': self.reference_data[column].min(),
                    'max': self.reference_data[column].max(),
                    'quantiles': self.reference_data[column].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                stats[column] = {
                    'value_counts': self.reference_data[column].value_counts().to_dict(),
                    'unique_count': self.reference_data[column].nunique()
                }
        return stats
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Detect drift using statistical tests."""
        drift_scores = {}
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
                
            if self.reference_data[column].dtype in ['int64', 'float64']:
                # Use Kolmogorov-Smirnov test for numerical features
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                drift_scores[column] = statistic
            else:
                # Use Chi-square test for categorical features
                ref_counts = self.reference_data[column].value_counts()
                curr_counts = current_data[column].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                if sum(curr_aligned) > 0:
                    statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
                    drift_scores[column] = statistic / sum(ref_aligned)  # Normalized
                else:
                    drift_scores[column] = 1.0  # Maximum drift
        
        return drift_scores
    
    def generate_drift_report(self, current_data: pd.DataFrame) -> Dict:
        """Generate comprehensive drift report using Evidently."""
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            
            # Generate drift report
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            return report.as_dict()
        except Exception as e:
            logger.error(f"Error generating drift report: {e}")
            return {}

class ModelMonitor:
    """Comprehensive model monitoring system."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = PrometheusMetrics()
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Prediction storage
        self.predictions_buffer = deque(maxlen=10000)
        self.performance_history = defaultdict(list)
        
        # Drift detection
        self.drift_detectors = {}
        
        # MLflow tracking
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        
    def record_prediction(self, prediction: ModelPrediction):
        """Record a model prediction for monitoring."""
        # Update Prometheus metrics
        self.metrics.prediction_counter.labels(
            model_name=prediction.model_name,
            status='success'
        ).inc()
        
        self.metrics.prediction_latency.labels(
            model_name=prediction.model_name
        ).observe(prediction.latency_ms / 1000.0)
        
        # Store in buffer
        self.predictions_buffer.append(prediction)
        
        # Store in Redis for real-time access
        prediction_data = {
            'model_name': prediction.model_name,
            'confidence': prediction.confidence,
            'latency_ms': prediction.latency_ms,
            'timestamp': prediction.timestamp.isoformat()
        }
        
        self.redis_client.lpush(
            f"predictions:{prediction.model_name}",
            json.dumps(prediction_data)
        )
        self.redis_client.ltrim(f"predictions:{prediction.model_name}", 0, 999)
    
    def record_error(self, model_name: str, error_type: str, error_message: str):
        """Record model errors."""
        self.metrics.prediction_counter.labels(
            model_name=model_name,
            status='error'
        ).inc()
        
        error_data = {
            'model_name': model_name,
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.redis_client.lpush("model_errors", json.dumps(error_data))
        self.redis_client.ltrim("model_errors", 0, 999)
    
    def update_model_metrics(self, model_name: str, metrics: ModelMetrics):
        """Update model performance metrics."""
        self.metrics.model_accuracy.labels(model_name=model_name).set(metrics.accuracy)
        
        # Store in performance history
        self.performance_history[model_name].append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Keep only last 1000 entries
        if len(self.performance_history[model_name]) > 1000:
            self.performance_history[model_name] = self.performance_history[model_name][-1000:]
    
    def check_data_drift(self, model_name: str, current_data: pd.DataFrame) -> Dict[str, float]:
        """Check for data drift in model inputs."""
        if model_name not in self.drift_detectors:
            logger.warning(f"No drift detector configured for model {model_name}")
            return {}
        
        drift_scores = self.drift_detectors[model_name].detect_drift(current_data)
        
        # Update Prometheus metrics
        for feature, score in drift_scores.items():
            self.metrics.model_drift_score.labels(
                model_name=model_name,
                feature=feature
            ).set(score)
        
        # Check if drift exceeds threshold
        high_drift_features = {
            feature: score for feature, score in drift_scores.items()
            if score > self.config.get('drift_threshold', 0.1)
        }
        
        if high_drift_features:
            self._trigger_drift_alert(model_name, high_drift_features)
        
        return drift_scores
    
    def _trigger_drift_alert(self, model_name: str, drift_features: Dict[str, float]):
        """Trigger alert for data drift."""
        alert_data = {
            'alert_type': 'data_drift',
            'model_name': model_name,
            'drift_features': drift_features,
            'timestamp': datetime.now().isoformat(),
            'severity': 'warning' if max(drift_features.values()) < 0.2 else 'critical'
        }
        
        # Store alert
        self.redis_client.lpush("alerts", json.dumps(alert_data))
        
        # Log alert
        logger.warning(f"Data drift detected for model {model_name}: {drift_features}")
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics.memory_usage.set(memory.used)
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                self.metrics.gpu_usage.labels(gpu_id=str(i)).set(gpu.load * 100)
                self.metrics.gpu_memory.labels(gpu_id=str(i)).set(gpu.memoryUsed * 1024 * 1024)
        except Exception as e:
            logger.debug(f"Could not collect GPU metrics: {e}")
    
    def record_business_metric(self, metric_type: str, value: float, labels: Dict[str, str] = None):
        """Record business-specific metrics."""
        labels = labels or {}
        
        if metric_type == 'articles_processed':
            self.metrics.news_articles_processed.labels(
                source=labels.get('source', 'unknown'),
                category=labels.get('category', 'unknown')
            ).inc(value)
        
        elif metric_type == 'summarization_quality':
            self.metrics.summarization_quality.labels(
                model_version=labels.get('model_version', 'unknown')
            ).observe(value)
        
        elif metric_type == 'user_engagement':
            self.metrics.user_engagement.labels(
                event_type=labels.get('event_type', 'unknown'),
                feature=labels.get('feature', 'unknown')
            ).inc(value)
    
    def get_model_health_score(self, model_name: str) -> float:
        """Calculate overall model health score."""
        if model_name not in self.performance_history:
            return 0.0
        
        recent_metrics = self.performance_history[model_name][-10:]  # Last 10 entries
        if not recent_metrics:
            return 0.0
        
        # Calculate weighted health score
        accuracy_scores = [m['metrics'].accuracy for m in recent_metrics]
        latency_scores = [1.0 / (1.0 + m['metrics'].latency_p95 / 1000.0) for m in recent_metrics]
        error_rates = [1.0 - m['metrics'].error_rate for m in recent_metrics]
        
        health_score = (
            0.4 * np.mean(accuracy_scores) +
            0.3 * np.mean(latency_scores) +
            0.3 * np.mean(error_rates)
        )
        
        return min(1.0, max(0.0, health_score))
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'system': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'alerts': []
        }
        
        # Model-specific reports
        for model_name in self.performance_history.keys():
            health_score = self.get_model_health_score(model_name)
            recent_predictions = len([
                p for p in self.predictions_buffer
                if p.model_name == model_name and
                p.timestamp > datetime.now() - timedelta(hours=1)
            ])
            
            report['models'][model_name] = {
                'health_score': health_score,
                'predictions_last_hour': recent_predictions,
                'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy'
            }
        
        # Recent alerts
        alert_keys = self.redis_client.lrange("alerts", 0, 9)
        for alert_key in alert_keys:
            try:
                alert_data = json.loads(alert_key)
                report['alerts'].append(alert_data)
            except json.JSONDecodeError:
                continue
        
        return report
    
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server."""
        start_http_server(port, registry=self.metrics.registry)
        logger.info(f"Metrics server started on port {port}")
    
    def start_monitoring_loop(self, interval: int = 30):
        """Start continuous monitoring loop."""
        logger.info(f"Starting monitoring loop with {interval}s interval")
        
        while True:
            try:
                self.update_system_metrics()
                
                # Log monitoring report every 5 minutes
                if int(time.time()) % 300 == 0:
                    report = self.generate_monitoring_report()
                    logger.info(f"Monitoring report: {json.dumps(report, indent=2)}")
                
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Monitoring loop stopped")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)

def main():
    """Main function to start monitoring system."""
    config = {
        'redis_host': os.getenv('REDIS_HOST', 'localhost'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        'mlflow_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        'drift_threshold': float(os.getenv('DRIFT_THRESHOLD', 0.1)),
        'metrics_port': int(os.getenv('METRICS_PORT', 8000))
    }
    
    # Initialize monitor
    monitor = ModelMonitor(config)
    
    # Start metrics server
    monitor.start_metrics_server(config['metrics_port'])
    
    # Start monitoring loop
    monitor.start_monitoring_loop()

if __name__ == "__main__":
    main()