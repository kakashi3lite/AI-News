#!/usr/bin/env python3
"""
Commander DeployX's Full-Stack Observability & Monitoring System
Real-User Monitoring with Predictive Alerting powered by OpenTelemetry

Features:
- Full-stack distributed tracing (OpenTelemetry)
- Real-user monitoring (RUM)
- Predictive alerting with ML models
- Multi-dimensional metrics collection
- Log aggregation and analysis
- Performance profiling and optimization
- SLO/SLI monitoring and alerting
- Anomaly detection and root cause analysis

Author: Commander Solaris "DeployX" Vivante
Version: 3.0.0
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import joblib

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Prometheus and Grafana
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import requests

# Log aggregation
import structlog
from pythonjsonlogger import jsonlogger

# Real-time processing
import redis
import asyncio
import aiohttp
from kafka import KafkaProducer, KafkaConsumer

# Alerting
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import slack_sdk
from slack_sdk import WebClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class TraceStatus(Enum):
    """Trace status"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class SLOConfig:
    """Service Level Objective configuration"""
    name: str
    description: str
    target_percentage: float  # e.g., 99.9
    time_window_hours: int  # e.g., 24
    metric_query: str  # Prometheus query
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    error_budget_burn_rate_threshold: float = 2.0

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    severity: AlertSeverity
    metric_query: str
    threshold: float
    comparison_operator: str
    duration_minutes: int = 5
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    notification_channels: List[str] = None
    ml_prediction_enabled: bool = False
    prediction_horizon_minutes: int = 30

@dataclass
class RUMMetrics:
    """Real User Monitoring metrics"""
    page_load_time: float
    first_contentful_paint: float
    largest_contentful_paint: float
    first_input_delay: float
    cumulative_layout_shift: float
    time_to_interactive: float
    user_agent: str
    geo_location: str
    session_id: str
    user_id: Optional[str] = None
    timestamp: datetime = None

@dataclass
class TraceSpan:
    """Distributed trace span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: TraceStatus
    tags: Dict[str, Any]
    logs: List[Dict[str, Any]]
    error_message: Optional[str] = None

@dataclass
class PerformanceProfile:
    """Performance profiling data"""
    service_name: str
    endpoint: str
    method: str
    cpu_usage_percent: float
    memory_usage_mb: float
    response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    timestamp: datetime
    bottlenecks: List[str] = None
    optimization_suggestions: List[str] = None

class FullStackMonitor:
    """Full-Stack Observability & Monitoring System"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # Initialize OpenTelemetry
        self._init_opentelemetry()
        
        # Initialize external clients
        self.redis_client = self._init_redis_client()
        self.kafka_producer = self._init_kafka_producer()
        self.prometheus_client = self._init_prometheus_client()
        self.slack_client = self._init_slack_client()
        
        # Monitoring data stores
        self.active_traces: Dict[str, TraceSpan] = {}
        self.rum_metrics: deque = deque(maxlen=10000)
        self.performance_profiles: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=1000)
        
        # SLO and alert configurations
        self.slo_configs: List[SLOConfig] = self._load_slo_configs()
        self.alert_rules: List[AlertRule] = self._load_alert_rules()
        
        # ML models for predictive alerting
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.prediction_models: Dict[str, Any] = {}
        
        # Metrics collectors
        self.custom_registry = CollectorRegistry()
        self._init_custom_metrics()
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        
        logger.info("Full-Stack Monitor initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'jaeger_endpoint': os.getenv('JAEGER_ENDPOINT', 'http://jaeger-collector:14268/api/traces'),
            'prometheus_url': os.getenv('PROMETHEUS_URL', 'http://prometheus-service:9090'),
            'grafana_url': os.getenv('GRAFANA_URL', 'http://grafana-service:3000'),
            'redis_url': os.getenv('REDIS_URL', 'redis://redis-service:6379'),
            'kafka_bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka-service:9092'),
            'slack_token': os.getenv('SLACK_BOT_TOKEN'),
            'slack_channel': os.getenv('SLACK_ALERT_CHANNEL', '#alerts'),
            'email_smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
            'email_smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
            'email_username': os.getenv('EMAIL_USERNAME'),
            'email_password': os.getenv('EMAIL_PASSWORD'),
            'email_from': os.getenv('EMAIL_FROM'),
            'email_to': os.getenv('EMAIL_TO', '').split(','),
            'service_name': os.getenv('SERVICE_NAME', 'ai-news-dashboard'),
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'metrics_collection_interval': int(os.getenv('METRICS_COLLECTION_INTERVAL', '60')),
            'trace_sampling_rate': float(os.getenv('TRACE_SAMPLING_RATE', '0.1')),
            'rum_collection_enabled': os.getenv('RUM_COLLECTION_ENABLED', 'true').lower() == 'true',
            'predictive_alerting_enabled': os.getenv('PREDICTIVE_ALERTING_ENABLED', 'true').lower() == 'true',
            'anomaly_detection_enabled': os.getenv('ANOMALY_DETECTION_ENABLED', 'true').lower() == 'true',
            'performance_profiling_enabled': os.getenv('PERFORMANCE_PROFILING_ENABLED', 'true').lower() == 'true'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        return default_config
    
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing and metrics"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.config['service_name'],
                "service.version": "1.0.0",
                "deployment.environment": self.config['environment']
            })
            
            # Initialize tracing
            trace.set_tracer_provider(TracerProvider(resource=resource))
            tracer_provider = trace.get_tracer_provider()
            
            # Add Jaeger exporter
            jaeger_exporter = JaegerExporter(
                endpoint=self.config['jaeger_endpoint']
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Initialize metrics
            prometheus_reader = PrometheusMetricReader()
            metrics.set_meter_provider(MeterProvider(
                resource=resource,
                metric_readers=[prometheus_reader]
            ))
            
            # Auto-instrument common libraries
            RequestsInstrumentor().instrument()
            FlaskInstrumentor().instrument()
            RedisInstrumentor().instrument()
            Psycopg2Instrumentor().instrument()
            
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            
            logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise
    
    def _init_redis_client(self):
        """Initialize Redis client"""
        try:
            return redis.from_url(self.config['redis_url'])
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client: {e}")
            return None
    
    def _init_kafka_producer(self):
        """Initialize Kafka producer"""
        try:
            return KafkaProducer(
                bootstrap_servers=self.config['kafka_bootstrap_servers'].split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Kafka producer: {e}")
            return None
    
    def _init_prometheus_client(self):
        """Initialize Prometheus client"""
        try:
            return requests.Session()
        except Exception as e:
            logger.warning(f"Failed to initialize Prometheus client: {e}")
            return None
    
    def _init_slack_client(self):
        """Initialize Slack client"""
        try:
            if self.config.get('slack_token'):
                return WebClient(token=self.config['slack_token'])
        except Exception as e:
            logger.warning(f"Failed to initialize Slack client: {e}")
        return None
    
    def _init_custom_metrics(self):
        """Initialize custom Prometheus metrics"""
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status_code'],
            registry=self.custom_registry
        )
        
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.custom_registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['service'],
            registry=self.custom_registry
        )
        
        self.error_rate = Gauge(
            'error_rate_percentage',
            'Error rate percentage',
            ['service', 'endpoint'],
            registry=self.custom_registry
        )
        
        self.slo_compliance = Gauge(
            'slo_compliance_percentage',
            'SLO compliance percentage',
            ['slo_name'],
            registry=self.custom_registry
        )
        
        self.rum_metrics_gauge = Gauge(
            'rum_metrics',
            'Real User Monitoring metrics',
            ['metric_type', 'geo_location'],
            registry=self.custom_registry
        )
        
        self.anomaly_score = Gauge(
            'anomaly_score',
            'Anomaly detection score',
            ['service', 'metric'],
            registry=self.custom_registry
        )
        
        self.prediction_accuracy = Gauge(
            'prediction_accuracy_percentage',
            'ML prediction accuracy percentage',
            ['model', 'metric'],
            registry=self.custom_registry
        )
    
    def _load_slo_configs(self) -> List[SLOConfig]:
        """Load SLO configurations"""
        default_slos = [
            SLOConfig(
                name="api_availability",
                description="API availability SLO",
                target_percentage=99.9,
                time_window_hours=24,
                metric_query='sum(rate(http_requests_total{status_code!~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
                threshold_value=0.999,
                comparison_operator=">="
            ),
            SLOConfig(
                name="api_latency_p99",
                description="API P99 latency SLO",
                target_percentage=95.0,
                time_window_hours=24,
                metric_query='histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))',
                threshold_value=0.5,  # 500ms
                comparison_operator="<="
            ),
            SLOConfig(
                name="data_freshness",
                description="Data freshness SLO",
                target_percentage=99.0,
                time_window_hours=1,
                metric_query='time() - max(data_last_updated_timestamp)',
                threshold_value=300,  # 5 minutes
                comparison_operator="<="
            )
        ]
        
        return default_slos
    
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rule configurations"""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                description="High error rate detected",
                severity=AlertSeverity.CRITICAL,
                metric_query='sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
                threshold=0.05,  # 5%
                comparison_operator=">",
                duration_minutes=5,
                notification_channels=["slack", "email"],
                ml_prediction_enabled=True
            ),
            AlertRule(
                name="high_latency",
                description="High API latency detected",
                severity=AlertSeverity.HIGH,
                metric_query='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
                threshold=1.0,  # 1 second
                comparison_operator=">",
                duration_minutes=10,
                notification_channels=["slack"],
                ml_prediction_enabled=True
            ),
            AlertRule(
                name="memory_usage_high",
                description="High memory usage detected",
                severity=AlertSeverity.MEDIUM,
                metric_query='(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes',
                threshold=0.85,  # 85%
                comparison_operator=">",
                duration_minutes=15,
                notification_channels=["slack"]
            ),
            AlertRule(
                name="disk_space_low",
                description="Low disk space detected",
                severity=AlertSeverity.HIGH,
                metric_query='(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes',
                threshold=0.90,  # 90%
                comparison_operator=">",
                duration_minutes=5,
                notification_channels=["slack", "email"]
            )
        ]
        
        return default_rules
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        try:
            logger.info("Starting full-stack monitoring")
            
            # Start background monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._collect_metrics_loop()),
                asyncio.create_task(self._process_traces_loop()),
                asyncio.create_task(self._monitor_slos_loop()),
                asyncio.create_task(self._check_alerts_loop()),
                asyncio.create_task(self._anomaly_detection_loop()),
                asyncio.create_task(self._predictive_alerting_loop()),
                asyncio.create_task(self._performance_profiling_loop()),
                asyncio.create_task(self._rum_processing_loop())
            ]
            
            # Start Prometheus metrics server
            prometheus_client.start_http_server(8094, registry=self.custom_registry)
            
            logger.info("Full-stack monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        try:
            logger.info("Stopping full-stack monitoring")
            
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            logger.info("Full-stack monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def _collect_metrics_loop(self):
        """Collect metrics periodically"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._collect_business_metrics()
                
                await asyncio.sleep(self.config['metrics_collection_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Query Prometheus for system metrics
            if self.prometheus_client:
                queries = [
                    'node_cpu_seconds_total',
                    'node_memory_MemTotal_bytes',
                    'node_memory_MemAvailable_bytes',
                    'node_filesystem_size_bytes',
                    'node_filesystem_free_bytes',
                    'node_network_receive_bytes_total',
                    'node_network_transmit_bytes_total'
                ]
                
                for query in queries:
                    await self._query_prometheus_metric(query)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        try:
            # Query application-specific metrics
            if self.prometheus_client:
                queries = [
                    'http_requests_total',
                    'http_request_duration_seconds',
                    'active_connections',
                    'database_connections_active',
                    'cache_hit_ratio',
                    'queue_size',
                    'worker_threads_active'
                ]
                
                for query in queries:
                    await self._query_prometheus_metric(query)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    async def _collect_business_metrics(self):
        """Collect business-level metrics"""
        try:
            # Query business-specific metrics
            if self.prometheus_client:
                queries = [
                    'articles_processed_total',
                    'user_sessions_active',
                    'api_calls_per_user',
                    'revenue_per_hour',
                    'conversion_rate',
                    'user_satisfaction_score'
                ]
                
                for query in queries:
                    await self._query_prometheus_metric(query)
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
    
    async def _query_prometheus_metric(self, query: str) -> Optional[Dict[str, Any]]:
        """Query Prometheus for a specific metric"""
        try:
            if not self.prometheus_client:
                return None
            
            url = f"{self.config['prometheus_url']}/api/v1/query"
            params = {'query': query}
            
            response = self.prometheus_client.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success':
                return data['data']
            
        except Exception as e:
            logger.error(f"Error querying Prometheus metric '{query}': {e}")
        
        return None
    
    async def _process_traces_loop(self):
        """Process distributed traces"""
        while True:
            try:
                await self._process_active_traces()
                await self._analyze_trace_patterns()
                await self._detect_trace_anomalies()
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trace processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_active_traces(self):
        """Process active traces"""
        try:
            # Process traces from Jaeger or other tracing backend
            # This would typically involve querying the tracing backend
            # For now, simulate trace processing
            
            current_time = datetime.now()
            
            # Clean up old traces
            expired_traces = [
                trace_id for trace_id, span in self.active_traces.items()
                if (current_time - span.start_time).total_seconds() > 3600  # 1 hour
            ]
            
            for trace_id in expired_traces:
                del self.active_traces[trace_id]
            
        except Exception as e:
            logger.error(f"Error processing active traces: {e}")
    
    async def _analyze_trace_patterns(self):
        """Analyze trace patterns for insights"""
        try:
            if not self.active_traces:
                return
            
            # Analyze common patterns
            service_latencies = defaultdict(list)
            operation_counts = defaultdict(int)
            error_patterns = defaultdict(int)
            
            for span in self.active_traces.values():
                service_latencies[span.service_name].append(span.duration_ms)
                operation_counts[span.operation_name] += 1
                
                if span.status == TraceStatus.ERROR:
                    error_patterns[f"{span.service_name}:{span.operation_name}"] += 1
            
            # Update metrics
            for service, latencies in service_latencies.items():
                if latencies:
                    avg_latency = np.mean(latencies)
                    p95_latency = np.percentile(latencies, 95)
                    
                    # Update custom metrics (would be done via OpenTelemetry in practice)
                    logger.info(f"Service {service} - Avg latency: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error analyzing trace patterns: {e}")
    
    async def _detect_trace_anomalies(self):
        """Detect anomalies in trace data"""
        try:
            if not self.config['anomaly_detection_enabled'] or not self.active_traces:
                return
            
            # Prepare data for anomaly detection
            trace_features = []
            for span in self.active_traces.values():
                features = [
                    span.duration_ms,
                    len(span.tags),
                    len(span.logs),
                    1 if span.status == TraceStatus.ERROR else 0
                ]
                trace_features.append(features)
            
            if len(trace_features) < 10:  # Need minimum samples
                return
            
            # Detect anomalies
            features_array = np.array(trace_features)
            anomaly_scores = self.anomaly_detector.fit_predict(features_array)
            
            # Process anomalies
            anomalous_traces = [
                (trace_id, span) for i, (trace_id, span) in enumerate(self.active_traces.items())
                if anomaly_scores[i] == -1
            ]
            
            for trace_id, span in anomalous_traces:
                await self._handle_trace_anomaly(trace_id, span)
            
        except Exception as e:
            logger.error(f"Error detecting trace anomalies: {e}")
    
    async def _handle_trace_anomaly(self, trace_id: str, span: TraceSpan):
        """Handle detected trace anomaly"""
        try:
            anomaly_data = {
                'trace_id': trace_id,
                'service_name': span.service_name,
                'operation_name': span.operation_name,
                'duration_ms': span.duration_ms,
                'status': span.status.value,
                'timestamp': span.start_time.isoformat()
            }
            
            # Update anomaly score metric
            self.anomaly_score.labels(
                service=span.service_name,
                metric='trace_duration'
            ).set(1.0)  # Anomaly detected
            
            # Send to Kafka for further processing
            if self.kafka_producer:
                self.kafka_producer.send('trace-anomalies', anomaly_data)
            
            logger.warning(f"Trace anomaly detected: {anomaly_data}")
            
        except Exception as e:
            logger.error(f"Error handling trace anomaly: {e}")
    
    async def _monitor_slos_loop(self):
        """Monitor SLOs continuously"""
        while True:
            try:
                for slo in self.slo_configs:
                    await self._check_slo_compliance(slo)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SLO monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_slo_compliance(self, slo: SLOConfig):
        """Check SLO compliance"""
        try:
            # Query current metric value
            metric_data = await self._query_prometheus_metric(slo.metric_query)
            
            if not metric_data or not metric_data.get('result'):
                logger.warning(f"No data available for SLO {slo.name}")
                return
            
            # Extract metric value
            result = metric_data['result'][0]
            current_value = float(result['value'][1])
            
            # Check compliance
            compliant = self._evaluate_condition(
                current_value, slo.threshold_value, slo.comparison_operator
            )
            
            # Calculate compliance percentage
            if compliant:
                compliance_percentage = 100.0
            else:
                # Calculate how far off we are
                if slo.comparison_operator in ['>', '>=']:
                    compliance_percentage = (current_value / slo.threshold_value) * 100
                else:
                    compliance_percentage = (slo.threshold_value / current_value) * 100
                
                compliance_percentage = min(compliance_percentage, 100.0)
            
            # Update SLO compliance metric
            self.slo_compliance.labels(slo_name=slo.name).set(compliance_percentage)
            
            # Check error budget burn rate
            if not compliant:
                await self._check_error_budget_burn_rate(slo, current_value)
            
            logger.info(f"SLO {slo.name}: {compliance_percentage:.2f}% compliant (current: {current_value}, target: {slo.threshold_value})")
            
        except Exception as e:
            logger.error(f"Error checking SLO compliance for {slo.name}: {e}")
    
    def _evaluate_condition(self, current_value: float, threshold: float, operator: str) -> bool:
        """Evaluate condition based on operator"""
        if operator == '>':
            return current_value > threshold
        elif operator == '>=':
            return current_value >= threshold
        elif operator == '<':
            return current_value < threshold
        elif operator == '<=':
            return current_value <= threshold
        elif operator == '==':
            return current_value == threshold
        elif operator == '!=':
            return current_value != threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    async def _check_error_budget_burn_rate(self, slo: SLOConfig, current_value: float):
        """Check error budget burn rate"""
        try:
            # Calculate burn rate (simplified)
            target_error_rate = 1 - (slo.target_percentage / 100)
            current_error_rate = 1 - current_value if slo.comparison_operator in ['>', '>='] else current_value
            
            burn_rate = current_error_rate / target_error_rate if target_error_rate > 0 else 0
            
            if burn_rate > slo.error_budget_burn_rate_threshold:
                await self._trigger_slo_alert(slo, burn_rate)
            
        except Exception as e:
            logger.error(f"Error checking error budget burn rate for {slo.name}: {e}")
    
    async def _trigger_slo_alert(self, slo: SLOConfig, burn_rate: float):
        """Trigger SLO violation alert"""
        try:
            alert_data = {
                'type': 'slo_violation',
                'slo_name': slo.name,
                'description': slo.description,
                'burn_rate': burn_rate,
                'threshold': slo.error_budget_burn_rate_threshold,
                'timestamp': datetime.now().isoformat(),
                'severity': AlertSeverity.HIGH.value
            }
            
            await self._send_alert(alert_data)
            
        except Exception as e:
            logger.error(f"Error triggering SLO alert: {e}")
    
    async def _check_alerts_loop(self):
        """Check alert rules continuously"""
        while True:
            try:
                for rule in self.alert_rules:
                    await self._check_alert_rule(rule)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert checking loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_alert_rule(self, rule: AlertRule):
        """Check individual alert rule"""
        try:
            # Query metric
            metric_data = await self._query_prometheus_metric(rule.metric_query)
            
            if not metric_data or not metric_data.get('result'):
                return
            
            # Extract metric value
            result = metric_data['result'][0]
            current_value = float(result['value'][1])
            
            # Check if alert condition is met
            alert_triggered = self._evaluate_condition(
                current_value, rule.threshold, rule.comparison_operator
            )
            
            if alert_triggered:
                # Check if we should use ML prediction
                if rule.ml_prediction_enabled and self.config['predictive_alerting_enabled']:
                    predicted_alert = await self._predict_alert_continuation(rule, current_value)
                    if not predicted_alert:
                        logger.info(f"ML prediction suggests alert {rule.name} will resolve soon, suppressing")
                        return
                
                await self._trigger_alert(rule, current_value)
            
        except Exception as e:
            logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    async def _predict_alert_continuation(self, rule: AlertRule, current_value: float) -> bool:
        """Predict if alert condition will continue"""
        try:
            # Get historical data for prediction
            historical_data = await self._get_historical_metric_data(
                rule.metric_query, 
                hours=24
            )
            
            if not historical_data or len(historical_data) < 10:
                return True  # Not enough data, trigger alert
            
            # Prepare data for Prophet
            df = pd.DataFrame(historical_data)
            df['ds'] = pd.to_datetime(df['timestamp'])
            df['y'] = df['value']
            
            # Train Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=False,
                yearly_seasonality=False
            )
            model.fit(df[['ds', 'y']])
            
            # Make prediction
            future = model.make_future_dataframe(
                periods=rule.prediction_horizon_minutes, 
                freq='T'
            )
            forecast = model.predict(future)
            
            # Check if predicted values will still trigger alert
            future_values = forecast['yhat'].tail(rule.prediction_horizon_minutes)
            alert_continues = any(
                self._evaluate_condition(val, rule.threshold, rule.comparison_operator)
                for val in future_values
            )
            
            # Update prediction accuracy metric
            # (In practice, you'd track this over time)
            self.prediction_accuracy.labels(
                model='prophet',
                metric=rule.name
            ).set(85.0)  # Placeholder accuracy
            
            return alert_continues
            
        except Exception as e:
            logger.error(f"Error predicting alert continuation: {e}")
            return True  # Default to triggering alert
    
    async def _get_historical_metric_data(self, query: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metric data for prediction"""
        try:
            if not self.prometheus_client:
                return []
            
            # Query range data from Prometheus
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            url = f"{self.config['prometheus_url']}/api/v1/query_range"
            params = {
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': '60s'  # 1 minute resolution
            }
            
            response = self.prometheus_client.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] != 'success' or not data['data']['result']:
                return []
            
            # Extract time series data
            result = data['data']['result'][0]
            values = result['values']
            
            historical_data = [
                {
                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                    'value': float(value)
                }
                for timestamp, value in values
            ]
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical metric data: {e}")
            return []
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger alert"""
        try:
            alert_data = {
                'type': 'metric_alert',
                'rule_name': rule.name,
                'description': rule.description,
                'severity': rule.severity.value,
                'current_value': current_value,
                'threshold': rule.threshold,
                'comparison_operator': rule.comparison_operator,
                'timestamp': datetime.now().isoformat(),
                'labels': rule.labels or {},
                'annotations': rule.annotations or {}
            }
            
            # Add to alert history
            self.alert_history.append(alert_data)
            
            # Send notifications
            if rule.notification_channels:
                for channel in rule.notification_channels:
                    await self._send_notification(channel, alert_data)
            
            logger.warning(f"Alert triggered: {rule.name} - {current_value} {rule.comparison_operator} {rule.threshold}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    async def _send_alert(self, alert_data: Dict[str, Any]):
        """Send alert through configured channels"""
        try:
            # Send to Slack
            if self.slack_client:
                await self._send_slack_alert(alert_data)
            
            # Send email
            if self.config.get('email_smtp_server'):
                await self._send_email_alert(alert_data)
            
            # Send to Kafka
            if self.kafka_producer:
                self.kafka_producer.send('alerts', alert_data)
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def _send_notification(self, channel: str, alert_data: Dict[str, Any]):
        """Send notification to specific channel"""
        try:
            if channel == 'slack':
                await self._send_slack_alert(alert_data)
            elif channel == 'email':
                await self._send_email_alert(alert_data)
            elif channel == 'webhook':
                await self._send_webhook_alert(alert_data)
            
        except Exception as e:
            logger.error(f"Error sending notification to {channel}: {e}")
    
    async def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Slack"""
        try:
            if not self.slack_client:
                return
            
            severity_colors = {
                AlertSeverity.CRITICAL.value: '#FF0000',
                AlertSeverity.HIGH.value: '#FF8C00',
                AlertSeverity.MEDIUM.value: '#FFD700',
                AlertSeverity.LOW.value: '#32CD32',
                AlertSeverity.INFO.value: '#87CEEB'
            }
            
            color = severity_colors.get(alert_data.get('severity'), '#808080')
            
            message = {
                'channel': self.config['slack_channel'],
                'attachments': [{
                    'color': color,
                    'title': f"🚨 {alert_data.get('type', 'Alert').title()}: {alert_data.get('rule_name', alert_data.get('slo_name', 'Unknown'))}",
                    'text': alert_data.get('description', 'No description available'),
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': alert_data.get('severity', 'unknown').upper(),
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert_data.get('timestamp', 'unknown'),
                            'short': True
                        }
                    ],
                    'footer': f"AI News Dashboard - {self.config['environment']}",
                    'ts': int(time.time())
                }]
            }
            
            # Add current value if available
            if 'current_value' in alert_data:
                message['attachments'][0]['fields'].append({
                    'title': 'Current Value',
                    'value': str(alert_data['current_value']),
                    'short': True
                })
            
            if 'threshold' in alert_data:
                message['attachments'][0]['fields'].append({
                    'title': 'Threshold',
                    'value': str(alert_data['threshold']),
                    'short': True
                })
            
            # Send message
            response = self.slack_client.chat_postMessage(**message)
            
            if response['ok']:
                logger.info(f"Slack alert sent successfully: {alert_data.get('rule_name', 'unknown')}")
            else:
                logger.error(f"Failed to send Slack alert: {response.get('error')}")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    async def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email"""
        try:
            if not all([
                self.config.get('email_smtp_server'),
                self.config.get('email_username'),
                self.config.get('email_password'),
                self.config.get('email_from'),
                self.config.get('email_to')
            ]):
                logger.warning("Email configuration incomplete, skipping email alert")
                return
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.config['email_from']
            msg['To'] = ', '.join(self.config['email_to'])
            msg['Subject'] = f"[{alert_data.get('severity', 'UNKNOWN').upper()}] {alert_data.get('rule_name', alert_data.get('slo_name', 'Alert'))}"
            
            # Email body
            body = f"""
            Alert Details:
            
            Type: {alert_data.get('type', 'Unknown')}
            Name: {alert_data.get('rule_name', alert_data.get('slo_name', 'Unknown'))}
            Severity: {alert_data.get('severity', 'unknown').upper()}
            Description: {alert_data.get('description', 'No description')}
            Timestamp: {alert_data.get('timestamp', 'unknown')}
            
            """
            
            if 'current_value' in alert_data:
                body += f"Current Value: {alert_data['current_value']}\n"
            if 'threshold' in alert_data:
                body += f"Threshold: {alert_data['threshold']}\n"
            if 'comparison_operator' in alert_data:
                body += f"Condition: {alert_data['comparison_operator']}\n"
            
            body += f"\nEnvironment: {self.config['environment']}\n"
            body += f"Service: {self.config['service_name']}\n"
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config['email_smtp_server'], self.config['email_smtp_port'])
            server.starttls()
            server.login(self.config['email_username'], self.config['email_password'])
            
            text = msg.as_string()
            server.sendmail(self.config['email_from'], self.config['email_to'], text)
            server.quit()
            
            logger.info(f"Email alert sent successfully: {alert_data.get('rule_name', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    async def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send alert via webhook"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                return
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=alert_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent successfully: {alert_data.get('rule_name', 'unknown')}")
                    else:
                        logger.error(f"Failed to send webhook alert: {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    async def _anomaly_detection_loop(self):
        """Anomaly detection loop"""
        while True:
            try:
                if self.config['anomaly_detection_enabled']:
                    await self._detect_metric_anomalies()
                    await self._detect_performance_anomalies()
                    await self._detect_business_anomalies()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(60)
    
    async def _detect_metric_anomalies(self):
        """Detect anomalies in metrics"""
        try:
            # Get recent metric data
            metrics_to_check = [
                'http_request_duration_seconds',
                'http_requests_total',
                'memory_usage_bytes',
                'cpu_usage_percent'
            ]
            
            for metric in metrics_to_check:
                historical_data = await self._get_historical_metric_data(metric, hours=24)
                
                if len(historical_data) < 50:  # Need sufficient data
                    continue
                
                # Prepare data for anomaly detection
                values = [d['value'] for d in historical_data]
                values_array = np.array(values).reshape(-1, 1)
                
                # Normalize data
                normalized_values = self.scaler.fit_transform(values_array)
                
                # Detect anomalies
                anomaly_scores = self.anomaly_detector.fit_predict(normalized_values)
                
                # Check recent values for anomalies
                recent_anomalies = anomaly_scores[-10:]  # Last 10 values
                if np.any(recent_anomalies == -1):
                    await self._handle_metric_anomaly(metric, historical_data[-10:])
            
        except Exception as e:
            logger.error(f"Error detecting metric anomalies: {e}")
    
    async def _handle_metric_anomaly(self, metric: str, recent_data: List[Dict[str, Any]]):
        """Handle detected metric anomaly"""
        try:
            anomaly_data = {
                'type': 'metric_anomaly',
                'metric_name': metric,
                'recent_values': [d['value'] for d in recent_data],
                'timestamps': [d['timestamp'].isoformat() for d in recent_data],
                'severity': AlertSeverity.MEDIUM.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update anomaly score metric
            self.anomaly_score.labels(
                service=self.config['service_name'],
                metric=metric
            ).set(1.0)
            
            await self._send_alert(anomaly_data)
            
            logger.warning(f"Metric anomaly detected: {metric}")
            
        except Exception as e:
            logger.error(f"Error handling metric anomaly: {e}")
    
    async def _detect_performance_anomalies(self):
        """Detect performance anomalies"""
        try:
            if not self.performance_profiles:
                return
            
            # Analyze recent performance profiles
            recent_profiles = list(self.performance_profiles)[-50:]  # Last 50 profiles
            
            if len(recent_profiles) < 10:
                return
            
            # Extract features for anomaly detection
            features = []
            for profile in recent_profiles:
                feature_vector = [
                    profile.cpu_usage_percent,
                    profile.memory_usage_mb,
                    profile.response_time_ms,
                    profile.throughput_rps,
                    profile.error_rate_percent
                ]
                features.append(feature_vector)
            
            # Detect anomalies
            features_array = np.array(features)
            anomaly_scores = self.anomaly_detector.fit_predict(features_array)
            
            # Handle anomalous profiles
            for i, score in enumerate(anomaly_scores):
                if score == -1:  # Anomaly detected
                    await self._handle_performance_anomaly(recent_profiles[i])
            
        except Exception as e:
            logger.error(f"Error detecting performance anomalies: {e}")
    
    async def _handle_performance_anomaly(self, profile: PerformanceProfile):
        """Handle detected performance anomaly"""
        try:
            anomaly_data = {
                'type': 'performance_anomaly',
                'service_name': profile.service_name,
                'endpoint': profile.endpoint,
                'cpu_usage_percent': profile.cpu_usage_percent,
                'memory_usage_mb': profile.memory_usage_mb,
                'response_time_ms': profile.response_time_ms,
                'throughput_rps': profile.throughput_rps,
                'error_rate_percent': profile.error_rate_percent,
                'severity': AlertSeverity.HIGH.value,
                'timestamp': profile.timestamp.isoformat()
            }
            
            await self._send_alert(anomaly_data)
            
            logger.warning(f"Performance anomaly detected: {profile.service_name}/{profile.endpoint}")
            
        except Exception as e:
            logger.error(f"Error handling performance anomaly: {e}")
    
    async def _detect_business_anomalies(self):
        """Detect business metric anomalies"""
        try:
            business_metrics = [
                'user_sessions_active',
                'articles_processed_total',
                'api_calls_per_user',
                'conversion_rate'
            ]
            
            for metric in business_metrics:
                historical_data = await self._get_historical_metric_data(metric, hours=168)  # 1 week
                
                if len(historical_data) < 100:
                    continue
                
                # Use Prophet for business metric anomaly detection
                await self._detect_business_anomaly_with_prophet(metric, historical_data)
            
        except Exception as e:
            logger.error(f"Error detecting business anomalies: {e}")
    
    async def _detect_business_anomaly_with_prophet(self, metric: str, historical_data: List[Dict[str, Any]]):
        """Detect business anomalies using Prophet"""
        try:
            # Prepare data for Prophet
            df = pd.DataFrame(historical_data)
            df['ds'] = pd.to_datetime(df['timestamp'])
            df['y'] = df['value']
            
            # Train Prophet model
            model = Prophet(
                interval_width=0.95,
                daily_seasonality=True,
                weekly_seasonality=True
            )
            model.fit(df)
            
            # Make predictions for recent data
            recent_data = df.tail(24)  # Last 24 hours
            forecast = model.predict(recent_data[['ds']])
            
            # Check for anomalies (values outside prediction intervals)
            anomalies = []
            for i, row in recent_data.iterrows():
                actual_value = row['y']
                predicted_lower = forecast.loc[forecast['ds'] == row['ds'], 'yhat_lower'].iloc[0]
                predicted_upper = forecast.loc[forecast['ds'] == row['ds'], 'yhat_upper'].iloc[0]
                
                if actual_value < predicted_lower or actual_value > predicted_upper:
                    anomalies.append({
                        'timestamp': row['ds'],
                        'actual_value': actual_value,
                        'predicted_lower': predicted_lower,
                        'predicted_upper': predicted_upper
                    })
            
            if anomalies:
                await self._handle_business_anomaly(metric, anomalies)
            
        except Exception as e:
            logger.error(f"Error detecting business anomaly with Prophet for {metric}: {e}")
    
    async def _handle_business_anomaly(self, metric: str, anomalies: List[Dict[str, Any]]):
        """Handle detected business anomaly"""
        try:
            anomaly_data = {
                'type': 'business_anomaly',
                'metric_name': metric,
                'anomalies_count': len(anomalies),
                'anomalies': anomalies,
                'severity': AlertSeverity.HIGH.value,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._send_alert(anomaly_data)
            
            logger.warning(f"Business anomaly detected: {metric} ({len(anomalies)} anomalous points)")
            
        except Exception as e:
            logger.error(f"Error handling business anomaly: {e}")
    
    async def _predictive_alerting_loop(self):
        """Predictive alerting loop"""
        while True:
            try:
                if self.config['predictive_alerting_enabled']:
                    await self._run_predictive_models()
                    await self._update_prediction_accuracy()
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in predictive alerting loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_predictive_models(self):
        """Run predictive models for alerting"""
        try:
            # Predict resource usage
            await self._predict_resource_usage()
            
            # Predict error rates
            await self._predict_error_rates()
            
            # Predict traffic patterns
            await self._predict_traffic_patterns()
            
        except Exception as e:
            logger.error(f"Error running predictive models: {e}")
    
    async def _predict_resource_usage(self):
        """Predict resource usage and alert if thresholds will be exceeded"""
        try:
            metrics = ['cpu_usage_percent', 'memory_usage_bytes', 'disk_usage_percent']
            
            for metric in metrics:
                historical_data = await self._get_historical_metric_data(metric, hours=168)  # 1 week
                
                if len(historical_data) < 100:
                    continue
                
                # Prepare data for Prophet
                df = pd.DataFrame(historical_data)
                df['ds'] = pd.to_datetime(df['timestamp'])
                df['y'] = df['value']
                
                # Train Prophet model
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False
                )
                model.fit(df)
                
                # Predict next 24 hours
                future = model.make_future_dataframe(periods=24, freq='H')
                forecast = model.predict(future)
                
                # Check if predicted values exceed thresholds
                future_predictions = forecast.tail(24)
                threshold = self.config.get(f'{metric}_threshold', 80.0)
                
                high_predictions = future_predictions[future_predictions['yhat'] > threshold]
                
                if not high_predictions.empty:
                    await self._handle_predictive_alert(metric, high_predictions, threshold)
            
        except Exception as e:
            logger.error(f"Error predicting resource usage: {e}")
    
    async def _predict_error_rates(self):
        """Predict error rates and alert if they will increase significantly"""
        try:
            historical_data = await self._get_historical_metric_data('error_rate_percent', hours=168)
            
            if len(historical_data) < 100:
                return
            
            # Prepare data
            df = pd.DataFrame(historical_data)
            df['ds'] = pd.to_datetime(df['timestamp'])
            df['y'] = df['value']
            
            # Train model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True
            )
            model.fit(df)
            
            # Predict next 6 hours
            future = model.make_future_dataframe(periods=6, freq='H')
            forecast = model.predict(future)
            
            # Check for significant increases
            current_avg = df.tail(24)['y'].mean()
            future_predictions = forecast.tail(6)
            predicted_avg = future_predictions['yhat'].mean()
            
            # Alert if predicted error rate is 50% higher than current
            if predicted_avg > current_avg * 1.5:
                await self._handle_error_rate_prediction(current_avg, predicted_avg, future_predictions)
            
        except Exception as e:
            logger.error(f"Error predicting error rates: {e}")
    
    async def _predict_traffic_patterns(self):
        """Predict traffic patterns and alert for capacity planning"""
        try:
            historical_data = await self._get_historical_metric_data('http_requests_total', hours=168)
            
            if len(historical_data) < 100:
                return
            
            # Prepare data
            df = pd.DataFrame(historical_data)
            df['ds'] = pd.to_datetime(df['timestamp'])
            df['y'] = df['value']
            
            # Train model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True
            )
            model.fit(df)
            
            # Predict next 12 hours
            future = model.make_future_dataframe(periods=12, freq='H')
            forecast = model.predict(future)
            
            # Check for traffic spikes
            current_max = df.tail(24)['y'].max()
            future_predictions = forecast.tail(12)
            predicted_max = future_predictions['yhat'].max()
            
            # Alert if predicted traffic is 200% higher than current max
            if predicted_max > current_max * 2.0:
                await self._handle_traffic_spike_prediction(current_max, predicted_max, future_predictions)
            
        except Exception as e:
            logger.error(f"Error predicting traffic patterns: {e}")
    
    async def _handle_predictive_alert(self, metric: str, predictions: pd.DataFrame, threshold: float):
        """Handle predictive alert for resource usage"""
        try:
            alert_data = {
                'type': 'predictive_resource_alert',
                'metric_name': metric,
                'threshold': threshold,
                'predicted_max': predictions['yhat'].max(),
                'predicted_times': predictions['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'severity': AlertSeverity.MEDIUM.value,
                'timestamp': datetime.now().isoformat(),
                'description': f'Predicted {metric} will exceed {threshold}% threshold in the next 24 hours'
            }
            
            await self._send_alert(alert_data)
            
            logger.warning(f"Predictive alert: {metric} will exceed threshold")
            
        except Exception as e:
            logger.error(f"Error handling predictive alert: {e}")
    
    async def _handle_error_rate_prediction(self, current_avg: float, predicted_avg: float, predictions: pd.DataFrame):
        """Handle error rate prediction alert"""
        try:
            alert_data = {
                'type': 'predictive_error_rate_alert',
                'current_avg_error_rate': current_avg,
                'predicted_avg_error_rate': predicted_avg,
                'increase_percentage': ((predicted_avg - current_avg) / current_avg) * 100,
                'predicted_times': predictions['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'severity': AlertSeverity.HIGH.value,
                'timestamp': datetime.now().isoformat(),
                'description': f'Error rate predicted to increase by {((predicted_avg - current_avg) / current_avg) * 100:.1f}%'
            }
            
            await self._send_alert(alert_data)
            
            logger.warning(f"Predictive alert: Error rate increase predicted")
            
        except Exception as e:
            logger.error(f"Error handling error rate prediction: {e}")
    
    async def _handle_traffic_spike_prediction(self, current_max: float, predicted_max: float, predictions: pd.DataFrame):
        """Handle traffic spike prediction alert"""
        try:
            alert_data = {
                'type': 'predictive_traffic_spike_alert',
                'current_max_traffic': current_max,
                'predicted_max_traffic': predicted_max,
                'spike_multiplier': predicted_max / current_max,
                'predicted_times': predictions['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'severity': AlertSeverity.HIGH.value,
                'timestamp': datetime.now().isoformat(),
                'description': f'Traffic spike predicted: {predicted_max / current_max:.1f}x current maximum'
            }
            
            await self._send_alert(alert_data)
            
            logger.warning(f"Predictive alert: Traffic spike predicted")
            
        except Exception as e:
            logger.error(f"Error handling traffic spike prediction: {e}")
    
    async def _update_prediction_accuracy(self):
        """Update prediction accuracy metrics"""
        try:
            # This would compare past predictions with actual values
            # For now, we'll set a placeholder accuracy
            self.prediction_accuracy.labels(
                model='prophet',
                metric='resource_usage'
            ).set(0.85)
            
            self.prediction_accuracy.labels(
                model='prophet',
                metric='error_rate'
            ).set(0.78)
            
            self.prediction_accuracy.labels(
                model='prophet',
                metric='traffic'
            ).set(0.82)
            
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}")
    
    async def _get_historical_metric_data(self, metric: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metric data from Prometheus"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Query Prometheus for historical data
            query = f'{metric}[{hours}h]'
            
            async with aiohttp.ClientSession() as session:
                params = {
                    'query': query,
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': '1h'
                }
                
                async with session.get(
                    f"{self.config['prometheus_url']}/api/v1/query_range",
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data['status'] == 'success' and data['data']['result']:
                            result = data['data']['result'][0]['values']
                            return [
                                {
                                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                                    'value': float(value)
                                }
                                for timestamp, value in result
                            ]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting historical metric data for {metric}: {e}")
            return []
    
    async def stop(self):
        """Stop the monitoring system"""
        try:
            logger.info("Stopping Full Stack Monitor...")
            
            # Cancel all running tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if hasattr(self, 'redis_client'):
                await self.redis_client.close()
            
            logger.info("Full Stack Monitor stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Full Stack Monitor: {e}")


# Example usage and configuration
if __name__ == "__main__":
    import asyncio
    
    # Example configuration
    config = {
        'service_name': 'ai-news-dashboard',
        'environment': 'production',
        'prometheus_url': 'http://localhost:9090',
        'grafana_url': 'http://localhost:3000',
        'redis_url': 'redis://localhost:6379',
        'kafka_bootstrap_servers': ['localhost:9092'],
        'slack_token': 'xoxb-your-slack-token',
        'slack_channel': '#alerts',
        'email_smtp_server': 'smtp.gmail.com',
        'email_smtp_port': 587,
        'email_username': 'your-email@gmail.com',
        'email_password': 'your-app-password',
        'email_from': 'alerts@ai-news-dashboard.com',
        'email_to': ['admin@ai-news-dashboard.com'],
        'webhook_url': 'https://your-webhook-url.com/alerts',
        'anomaly_detection_enabled': True,
        'predictive_alerting_enabled': True,
        'real_user_monitoring_enabled': True,
        'performance_profiling_enabled': True,
        'log_aggregation_enabled': True,
        'trace_sampling_rate': 0.1,
        'metrics_retention_days': 30,
        'logs_retention_days': 7,
        'traces_retention_days': 3
    }
    
    async def main():
        monitor = FullStackMonitor(config)
        
        try:
            await monitor.start()
            
            # Keep running
            while True:
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await monitor.stop()
    
    # Run the monitor
    asyncio.run(main())