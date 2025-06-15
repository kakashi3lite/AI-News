#!/usr/bin/env python3
"""
Full-Stack Observability System

This module provides comprehensive observability across the entire deployment pipeline,
including metrics collection, distributed tracing, log aggregation, alerting,
and real-user monitoring.

Features:
- OpenTelemetry integration for distributed tracing
- Prometheus metrics collection and analysis
- Grafana dashboard automation
- ELK/Loki log aggregation
- Real-user monitoring (RUM)
- SLI/SLO tracking and alerting
- Anomaly detection and predictive alerting
- Performance profiling and optimization

Author: Commander Solaris "DeployX" Vivante
"""

import asyncio
import logging
import time
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import subprocess
from pathlib import Path
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
import statistics
import warnings
warnings.filterwarnings('ignore')

# Observability libraries (would be imported in real implementation)
# from opentelemetry import trace, metrics
# from opentelemetry.exporter.prometheus import PrometheusMetricReader
# from opentelemetry.exporter.jaeger.thrift import JaegerExporter
# from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
# import grafana_api
# from elasticsearch import Elasticsearch
# import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertStatus(Enum):
    """Alert status"""
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    ACKNOWLEDGED = "acknowledged"

class TraceStatus(Enum):
    """Trace status"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class SLOStatus(Enum):
    """SLO status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACHED = "breached"

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    type: MetricType
    description: str
    labels: List[str]
    unit: str
    aggregation: str  # sum, avg, max, min, p95, p99
    retention_days: int
    alert_thresholds: Dict[str, float]

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
    error_message: Optional[str]

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    service: str
    component: str
    message: str
    trace_id: Optional[str]
    span_id: Optional[str]
    labels: Dict[str, str]
    fields: Dict[str, Any]
    source: str

@dataclass
class Alert:
    """Alert definition and state"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    query: str
    threshold: float
    comparison: str  # >, <, >=, <=, ==, !=
    duration: str  # 5m, 10m, 1h
    labels: Dict[str, str]
    annotations: Dict[str, str]
    fired_at: Optional[datetime]
    resolved_at: Optional[datetime]
    notification_channels: List[str]
    runbook_url: Optional[str]

@dataclass
class SLI:
    """Service Level Indicator"""
    name: str
    description: str
    query: str
    unit: str
    good_events_query: str
    total_events_query: str
    target_value: float
    current_value: float
    trend: str  # improving, stable, degrading

@dataclass
class SLO:
    """Service Level Objective"""
    name: str
    description: str
    sli: SLI
    target_percentage: float
    time_window: str  # 1d, 7d, 30d
    current_percentage: float
    error_budget_remaining: float
    status: SLOStatus
    burn_rate: float
    alerts: List[Alert]

@dataclass
class Dashboard:
    """Grafana dashboard definition"""
    id: str
    title: str
    description: str
    tags: List[str]
    panels: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    time_range: Dict[str, str]
    refresh_interval: str
    created_at: datetime
    updated_at: datetime

@dataclass
class RUMMetric:
    """Real User Monitoring metric"""
    session_id: str
    user_id: Optional[str]
    page_url: str
    user_agent: str
    country: str
    city: str
    connection_type: str
    page_load_time_ms: float
    first_contentful_paint_ms: float
    largest_contentful_paint_ms: float
    cumulative_layout_shift: float
    first_input_delay_ms: float
    time_to_interactive_ms: float
    errors: List[Dict[str, Any]]
    custom_metrics: Dict[str, float]
    timestamp: datetime

class FullStackObserver:
    """Full-Stack Observability System"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Full-Stack Observer"""
        self.config = self._load_config(config_path)
        
        # Metrics and tracing
        self.metrics_registry = {}
        self.active_traces = {}
        self.trace_history = []
        
        # Alerting
        self.alerts = {}
        self.alert_rules = []
        self.notification_channels = {}
        
        # SLI/SLO tracking
        self.slos = {}
        self.sli_history = {}
        
        # Dashboards
        self.dashboards = {}
        
        # Log aggregation
        self.log_buffer = []
        self.log_processors = []
        
        # RUM data
        self.rum_sessions = {}
        self.rum_metrics = []
        
        # Anomaly detection
        self.anomaly_detectors = {}
        self.baseline_metrics = {}
        
        logger.info("Full-Stack Observer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load observability configuration"""
        default_config = {
            "metrics": {
                "prometheus": {
                    "enabled": True,
                    "endpoint": "http://localhost:9090",
                    "scrape_interval": "15s",
                    "retention": "15d"
                },
                "custom_metrics": {
                    "enabled": True,
                    "export_interval": "10s"
                }
            },
            "tracing": {
                "enabled": True,
                "jaeger": {
                    "endpoint": "http://localhost:14268/api/traces",
                    "service_name": "ai-news-dashboard"
                },
                "sampling_rate": 0.1,
                "max_spans_per_trace": 1000
            },
            "logging": {
                "enabled": True,
                "elasticsearch": {
                    "enabled": False,
                    "hosts": ["localhost:9200"],
                    "index_pattern": "logs-*"
                },
                "loki": {
                    "enabled": True,
                    "endpoint": "http://localhost:3100"
                },
                "log_level": "INFO",
                "structured_logging": True
            },
            "alerting": {
                "enabled": True,
                "alertmanager": {
                    "endpoint": "http://localhost:9093"
                },
                "notification_channels": {
                    "slack": {
                        "enabled": True,
                        "webhook_url": "https://hooks.slack.com/..."
                    },
                    "email": {
                        "enabled": True,
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587
                    },
                    "pagerduty": {
                        "enabled": False,
                        "integration_key": ""
                    }
                },
                "default_severity": "medium",
                "auto_resolve": True
            },
            "dashboards": {
                "grafana": {
                    "enabled": True,
                    "endpoint": "http://localhost:3000",
                    "api_key": "",
                    "auto_provision": True
                },
                "custom_dashboards": True
            },
            "slo": {
                "enabled": True,
                "default_targets": {
                    "availability": 99.9,
                    "latency_p95": 200,
                    "error_rate": 0.1
                },
                "burn_rate_alerts": True
            },
            "rum": {
                "enabled": True,
                "sampling_rate": 0.1,
                "session_timeout": 1800,
                "performance_budget": {
                    "page_load_time_ms": 3000,
                    "first_contentful_paint_ms": 1500,
                    "largest_contentful_paint_ms": 2500
                }
            },
            "anomaly_detection": {
                "enabled": True,
                "algorithms": ["isolation_forest", "statistical"],
                "sensitivity": 0.1,
                "baseline_period_days": 7
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    async def initialize(self):
        """Initialize the observability system"""
        logger.info("Initializing Full-Stack Observer...")
        
        try:
            # Initialize metrics collection
            await self._initialize_metrics()
            
            # Initialize distributed tracing
            await self._initialize_tracing()
            
            # Initialize log aggregation
            await self._initialize_logging()
            
            # Initialize alerting
            await self._initialize_alerting()
            
            # Initialize dashboards
            await self._initialize_dashboards()
            
            # Initialize SLO tracking
            await self._initialize_slo_tracking()
            
            # Initialize RUM
            await self._initialize_rum()
            
            # Initialize anomaly detection
            await self._initialize_anomaly_detection()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Full-Stack Observer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Full-Stack Observer: {e}")
            raise
    
    async def _initialize_metrics(self):
        """Initialize metrics collection"""
        if not self.config["metrics"]["prometheus"]["enabled"]:
            return
        
        logger.info("Initializing metrics collection...")
        
        # Define core metrics
        core_metrics = [
            MetricDefinition(
                name="http_requests_total",
                type=MetricType.COUNTER,
                description="Total HTTP requests",
                labels=["method", "endpoint", "status_code"],
                unit="requests",
                aggregation="sum",
                retention_days=30,
                alert_thresholds={"error_rate": 0.05}
            ),
            MetricDefinition(
                name="http_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="HTTP request duration",
                labels=["method", "endpoint"],
                unit="seconds",
                aggregation="p95",
                retention_days=30,
                alert_thresholds={"p95_latency": 0.5}
            ),
            MetricDefinition(
                name="deployment_status",
                type=MetricType.GAUGE,
                description="Deployment status (1=success, 0=failure)",
                labels=["service", "version", "environment"],
                unit="status",
                aggregation="last",
                retention_days=90,
                alert_thresholds={"failure": 0}
            ),
            MetricDefinition(
                name="canary_analysis_score",
                type=MetricType.GAUGE,
                description="Canary analysis score",
                labels=["service", "version"],
                unit="score",
                aggregation="avg",
                retention_days=30,
                alert_thresholds={"low_score": 0.7}
            )
        ]
        
        for metric in core_metrics:
            self.metrics_registry[metric.name] = metric
        
        logger.info(f"Initialized {len(core_metrics)} core metrics")
    
    async def _initialize_tracing(self):
        """Initialize distributed tracing"""
        if not self.config["tracing"]["enabled"]:
            return
        
        logger.info("Initializing distributed tracing...")
        
        # In real implementation, configure OpenTelemetry
        # tracer = trace.get_tracer(__name__)
        
        logger.info("Distributed tracing initialized")
    
    async def _initialize_logging(self):
        """Initialize log aggregation"""
        if not self.config["logging"]["enabled"]:
            return
        
        logger.info("Initializing log aggregation...")
        
        # Initialize log processors
        self.log_processors = [
            self._process_error_logs,
            self._process_performance_logs,
            self._process_security_logs
        ]
        
        logger.info("Log aggregation initialized")
    
    async def _initialize_alerting(self):
        """Initialize alerting system"""
        if not self.config["alerting"]["enabled"]:
            return
        
        logger.info("Initializing alerting system...")
        
        # Define core alert rules
        core_alerts = [
            Alert(
                id="high-error-rate",
                name="High Error Rate",
                description="Error rate is above threshold",
                severity=AlertSeverity.HIGH,
                status=AlertStatus.RESOLVED,
                query="rate(http_requests_total{status_code=~'5..'}[5m]) > 0.05",
                threshold=0.05,
                comparison=">",
                duration="5m",
                labels={"team": "platform", "service": "api"},
                annotations={"summary": "High error rate detected"},
                fired_at=None,
                resolved_at=None,
                notification_channels=["slack", "email"],
                runbook_url="https://runbooks.example.com/high-error-rate"
            ),
            Alert(
                id="high-latency",
                name="High Latency",
                description="95th percentile latency is above threshold",
                severity=AlertSeverity.MEDIUM,
                status=AlertStatus.RESOLVED,
                query="histogram_quantile(0.95, http_request_duration_seconds) > 0.5",
                threshold=0.5,
                comparison=">",
                duration="10m",
                labels={"team": "platform", "service": "api"},
                annotations={"summary": "High latency detected"},
                fired_at=None,
                resolved_at=None,
                notification_channels=["slack"],
                runbook_url="https://runbooks.example.com/high-latency"
            ),
            Alert(
                id="deployment-failure",
                name="Deployment Failure",
                description="Deployment has failed",
                severity=AlertSeverity.CRITICAL,
                status=AlertStatus.RESOLVED,
                query="deployment_status == 0",
                threshold=0,
                comparison="==",
                duration="1m",
                labels={"team": "platform", "service": "deployment"},
                annotations={"summary": "Deployment failure detected"},
                fired_at=None,
                resolved_at=None,
                notification_channels=["slack", "email", "pagerduty"],
                runbook_url="https://runbooks.example.com/deployment-failure"
            )
        ]
        
        for alert in core_alerts:
            self.alerts[alert.id] = alert
        
        logger.info(f"Initialized {len(core_alerts)} alert rules")
    
    async def _initialize_dashboards(self):
        """Initialize dashboards"""
        if not self.config["dashboards"]["grafana"]["enabled"]:
            return
        
        logger.info("Initializing dashboards...")
        
        # Create core dashboards
        await self._create_deployment_dashboard()
        await self._create_application_dashboard()
        await self._create_infrastructure_dashboard()
        await self._create_slo_dashboard()
        
        logger.info("Dashboards initialized")
    
    async def _create_deployment_dashboard(self):
        """Create deployment monitoring dashboard"""
        dashboard = Dashboard(
            id="deployment-monitoring",
            title="Deployment Monitoring",
            description="Monitor deployment pipeline and canary analysis",
            tags=["deployment", "canary", "pipeline"],
            panels=[
                {
                    "title": "Deployment Status",
                    "type": "stat",
                    "targets": [{"expr": "deployment_status"}],
                    "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}}}
                },
                {
                    "title": "Canary Analysis Score",
                    "type": "timeseries",
                    "targets": [{"expr": "canary_analysis_score"}]
                },
                {
                    "title": "Deployment Duration",
                    "type": "timeseries",
                    "targets": [{"expr": "deployment_duration_seconds"}]
                }
            ],
            variables=[],
            time_range={"from": "now-1h", "to": "now"},
            refresh_interval="30s",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.dashboards[dashboard.id] = dashboard
        logger.info(f"Created dashboard: {dashboard.title}")
    
    async def _create_application_dashboard(self):
        """Create application monitoring dashboard"""
        dashboard = Dashboard(
            id="application-monitoring",
            title="Application Monitoring",
            description="Monitor application performance and errors",
            tags=["application", "performance", "errors"],
            panels=[
                {
                    "title": "Request Rate",
                    "type": "timeseries",
                    "targets": [{"expr": "rate(http_requests_total[5m])"}]
                },
                {
                    "title": "Error Rate",
                    "type": "timeseries",
                    "targets": [{"expr": "rate(http_requests_total{status_code=~'5..'}[5m])"}]
                },
                {
                    "title": "Response Time (P95)",
                    "type": "timeseries",
                    "targets": [{"expr": "histogram_quantile(0.95, http_request_duration_seconds)"}]
                }
            ],
            variables=[{"name": "service", "type": "query", "query": "label_values(service)"}],
            time_range={"from": "now-1h", "to": "now"},
            refresh_interval="15s",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.dashboards[dashboard.id] = dashboard
        logger.info(f"Created dashboard: {dashboard.title}")
    
    async def _create_infrastructure_dashboard(self):
        """Create infrastructure monitoring dashboard"""
        dashboard = Dashboard(
            id="infrastructure-monitoring",
            title="Infrastructure Monitoring",
            description="Monitor infrastructure resources and health",
            tags=["infrastructure", "resources", "health"],
            panels=[
                {
                    "title": "CPU Usage",
                    "type": "timeseries",
                    "targets": [{"expr": "cpu_usage_percent"}]
                },
                {
                    "title": "Memory Usage",
                    "type": "timeseries",
                    "targets": [{"expr": "memory_usage_percent"}]
                },
                {
                    "title": "Pod Status",
                    "type": "stat",
                    "targets": [{"expr": "kube_pod_status_phase"}]
                }
            ],
            variables=[{"name": "cluster", "type": "query", "query": "label_values(cluster)"}],
            time_range={"from": "now-1h", "to": "now"},
            refresh_interval="30s",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.dashboards[dashboard.id] = dashboard
        logger.info(f"Created dashboard: {dashboard.title}")
    
    async def _create_slo_dashboard(self):
        """Create SLO monitoring dashboard"""
        dashboard = Dashboard(
            id="slo-monitoring",
            title="SLO Monitoring",
            description="Monitor Service Level Objectives and error budgets",
            tags=["slo", "sli", "error-budget"],
            panels=[
                {
                    "title": "SLO Compliance",
                    "type": "stat",
                    "targets": [{"expr": "slo_compliance_percentage"}]
                },
                {
                    "title": "Error Budget Remaining",
                    "type": "bargauge",
                    "targets": [{"expr": "error_budget_remaining_percentage"}]
                },
                {
                    "title": "Burn Rate",
                    "type": "timeseries",
                    "targets": [{"expr": "slo_burn_rate"}]
                }
            ],
            variables=[{"name": "slo", "type": "query", "query": "label_values(slo_name)"}],
            time_range={"from": "now-7d", "to": "now"},
            refresh_interval="1m",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.dashboards[dashboard.id] = dashboard
        logger.info(f"Created dashboard: {dashboard.title}")
    
    async def _initialize_slo_tracking(self):
        """Initialize SLO tracking"""
        if not self.config["slo"]["enabled"]:
            return
        
        logger.info("Initializing SLO tracking...")
        
        # Define core SLOs
        availability_sli = SLI(
            name="availability",
            description="Service availability",
            query="up",
            unit="percentage",
            good_events_query="http_requests_total{status_code!~'5..'}",
            total_events_query="http_requests_total",
            target_value=99.9,
            current_value=99.95,
            trend="stable"
        )
        
        availability_slo = SLO(
            name="availability-slo",
            description="99.9% availability over 30 days",
            sli=availability_sli,
            target_percentage=99.9,
            time_window="30d",
            current_percentage=99.95,
            error_budget_remaining=50.0,
            status=SLOStatus.HEALTHY,
            burn_rate=0.1,
            alerts=[]
        )
        
        self.slos[availability_slo.name] = availability_slo
        
        logger.info(f"Initialized {len(self.slos)} SLOs")
    
    async def _initialize_rum(self):
        """Initialize Real User Monitoring"""
        if not self.config["rum"]["enabled"]:
            return
        
        logger.info("Initializing Real User Monitoring...")
        
        # RUM is typically initialized on the client side
        # Here we set up the backend to receive RUM data
        
        logger.info("Real User Monitoring initialized")
    
    async def _initialize_anomaly_detection(self):
        """Initialize anomaly detection"""
        if not self.config["anomaly_detection"]["enabled"]:
            return
        
        logger.info("Initializing anomaly detection...")
        
        # Initialize anomaly detectors for key metrics
        # In real implementation, use ML models
        
        logger.info("Anomaly detection initialized")
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        logger.info("Starting background monitoring tasks...")
        
        # Start metric collection
        asyncio.create_task(self._metric_collection_loop())
        
        # Start alert evaluation
        asyncio.create_task(self._alert_evaluation_loop())
        
        # Start SLO calculation
        asyncio.create_task(self._slo_calculation_loop())
        
        # Start log processing
        asyncio.create_task(self._log_processing_loop())
        
        # Start anomaly detection
        asyncio.create_task(self._anomaly_detection_loop())
        
        logger.info("Background monitoring tasks started")
    
    async def _metric_collection_loop(self):
        """Background loop for metric collection"""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(15)  # Collect every 15 seconds
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self):
        """Collect metrics from various sources"""
        # In real implementation, scrape Prometheus, query APIs, etc.
        # For demo, simulate metric collection
        
        import random
        timestamp = datetime.now()
        
        # Simulate HTTP metrics
        self._record_metric("http_requests_total", random.randint(100, 1000), 
                          {"method": "GET", "endpoint": "/api/news", "status_code": "200"})
        
        # Simulate deployment metrics
        self._record_metric("deployment_status", 1 if random.random() > 0.05 else 0,
                          {"service": "api", "version": "v2.0.0", "environment": "prod"})
        
        # Simulate canary metrics
        self._record_metric("canary_analysis_score", random.uniform(0.7, 1.0),
                          {"service": "api", "version": "v2.0.0"})
    
    def _record_metric(self, name: str, value: float, labels: Dict[str, str]):
        """Record a metric value"""
        if name not in self.metrics_registry:
            return
        
        # In real implementation, send to Prometheus or other TSDB
        # For demo, store in memory
        key = f"{name}_{hash(str(sorted(labels.items())))}"
        if key not in self.baseline_metrics:
            self.baseline_metrics[key] = []
        
        self.baseline_metrics[key].append({
            "timestamp": datetime.now(),
            "value": value,
            "labels": labels
        })
        
        # Keep only recent data
        cutoff = datetime.now() - timedelta(hours=24)
        self.baseline_metrics[key] = [
            m for m in self.baseline_metrics[key] if m["timestamp"] > cutoff
        ]
    
    async def _alert_evaluation_loop(self):
        """Background loop for alert evaluation"""
        while True:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_alerts(self):
        """Evaluate alert rules"""
        for alert_id, alert in self.alerts.items():
            try:
                # Evaluate alert condition
                triggered = await self._evaluate_alert_condition(alert)
                
                if triggered and alert.status == AlertStatus.RESOLVED:
                    # Fire alert
                    await self._fire_alert(alert)
                elif not triggered and alert.status == AlertStatus.FIRING:
                    # Resolve alert
                    await self._resolve_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert {alert_id}: {e}")
    
    async def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate if alert condition is met"""
        # In real implementation, query Prometheus with alert.query
        # For demo, simulate alert evaluation
        
        import random
        
        if "error-rate" in alert.id:
            return random.random() < 0.1  # 10% chance of high error rate
        elif "latency" in alert.id:
            return random.random() < 0.05  # 5% chance of high latency
        elif "deployment" in alert.id:
            return random.random() < 0.02  # 2% chance of deployment failure
        
        return False
    
    async def _fire_alert(self, alert: Alert):
        """Fire an alert"""
        logger.warning(f"ALERT FIRED: {alert.name} - {alert.description}")
        
        alert.status = AlertStatus.FIRING
        alert.fired_at = datetime.now()
        alert.resolved_at = None
        
        # Send notifications
        await self._send_alert_notifications(alert)
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        logger.info(f"ALERT RESOLVED: {alert.name}")
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Send resolution notifications
        await self._send_alert_notifications(alert, resolved=True)
    
    async def _send_alert_notifications(self, alert: Alert, resolved: bool = False):
        """Send alert notifications"""
        action = "RESOLVED" if resolved else "FIRED"
        message = f"Alert {action}: {alert.name} - {alert.description}"
        
        for channel in alert.notification_channels:
            try:
                if channel == "slack":
                    await self._send_slack_notification(message, alert)
                elif channel == "email":
                    await self._send_email_notification(message, alert)
                elif channel == "pagerduty":
                    await self._send_pagerduty_notification(message, alert)
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")
    
    async def _send_slack_notification(self, message: str, alert: Alert):
        """Send Slack notification"""
        # In real implementation, send to Slack webhook
        logger.info(f"Slack notification: {message}")
    
    async def _send_email_notification(self, message: str, alert: Alert):
        """Send email notification"""
        # In real implementation, send email via SMTP
        logger.info(f"Email notification: {message}")
    
    async def _send_pagerduty_notification(self, message: str, alert: Alert):
        """Send PagerDuty notification"""
        # In real implementation, send to PagerDuty API
        logger.info(f"PagerDuty notification: {message}")
    
    async def _slo_calculation_loop(self):
        """Background loop for SLO calculation"""
        while True:
            try:
                await self._calculate_slos()
                await asyncio.sleep(300)  # Calculate every 5 minutes
            except Exception as e:
                logger.error(f"SLO calculation error: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_slos(self):
        """Calculate SLO compliance and error budgets"""
        for slo_name, slo in self.slos.items():
            try:
                # Calculate current SLI value
                current_value = await self._calculate_sli_value(slo.sli)
                slo.sli.current_value = current_value
                
                # Calculate SLO compliance
                compliance = await self._calculate_slo_compliance(slo)
                slo.current_percentage = compliance
                
                # Calculate error budget
                error_budget = await self._calculate_error_budget(slo)
                slo.error_budget_remaining = error_budget
                
                # Calculate burn rate
                burn_rate = await self._calculate_burn_rate(slo)
                slo.burn_rate = burn_rate
                
                # Update SLO status
                slo.status = self._determine_slo_status(slo)
                
                logger.debug(f"SLO {slo_name}: {compliance:.2f}% compliance, {error_budget:.2f}% error budget")
                
            except Exception as e:
                logger.error(f"Error calculating SLO {slo_name}: {e}")
    
    async def _calculate_sli_value(self, sli: SLI) -> float:
        """Calculate current SLI value"""
        # In real implementation, query metrics backend
        # For demo, simulate SLI calculation
        import random
        return random.uniform(99.0, 99.99)
    
    async def _calculate_slo_compliance(self, slo: SLO) -> float:
        """Calculate SLO compliance percentage"""
        # In real implementation, calculate based on time window
        # For demo, simulate compliance calculation
        import random
        return random.uniform(99.5, 99.99)
    
    async def _calculate_error_budget(self, slo: SLO) -> float:
        """Calculate remaining error budget percentage"""
        # Error budget = (1 - target) * 100
        # Remaining = budget - consumed
        import random
        return random.uniform(20.0, 80.0)
    
    async def _calculate_burn_rate(self, slo: SLO) -> float:
        """Calculate error budget burn rate"""
        # Burn rate = rate of error budget consumption
        import random
        return random.uniform(0.1, 2.0)
    
    def _determine_slo_status(self, slo: SLO) -> SLOStatus:
        """Determine SLO status based on compliance and burn rate"""
        if slo.current_percentage < slo.target_percentage:
            return SLOStatus.BREACHED
        elif slo.error_budget_remaining < 10:
            return SLOStatus.CRITICAL
        elif slo.error_budget_remaining < 25:
            return SLOStatus.WARNING
        else:
            return SLOStatus.HEALTHY
    
    async def _log_processing_loop(self):
        """Background loop for log processing"""
        while True:
            try:
                await self._process_logs()
                await asyncio.sleep(10)  # Process every 10 seconds
            except Exception as e:
                logger.error(f"Log processing error: {e}")
                await asyncio.sleep(60)
    
    async def _process_logs(self):
        """Process accumulated logs"""
        if not self.log_buffer:
            return
        
        # Process logs in batch
        logs_to_process = self.log_buffer[:100]  # Process up to 100 logs
        self.log_buffer = self.log_buffer[100:]
        
        for log_entry in logs_to_process:
            for processor in self.log_processors:
                try:
                    await processor(log_entry)
                except Exception as e:
                    logger.error(f"Log processor error: {e}")
    
    async def _process_error_logs(self, log_entry: LogEntry):
        """Process error logs"""
        if log_entry.level in ["ERROR", "CRITICAL"]:
            # Extract error patterns, create alerts if needed
            logger.debug(f"Processing error log: {log_entry.message}")
    
    async def _process_performance_logs(self, log_entry: LogEntry):
        """Process performance logs"""
        if "performance" in log_entry.labels:
            # Extract performance metrics
            logger.debug(f"Processing performance log: {log_entry.message}")
    
    async def _process_security_logs(self, log_entry: LogEntry):
        """Process security logs"""
        if "security" in log_entry.labels:
            # Analyze for security threats
            logger.debug(f"Processing security log: {log_entry.message}")
    
    async def _anomaly_detection_loop(self):
        """Background loop for anomaly detection"""
        while True:
            try:
                await self._detect_anomalies()
                await asyncio.sleep(60)  # Detect every minute
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(300)
    
    async def _detect_anomalies(self):
        """Detect anomalies in metrics"""
        for metric_key, data_points in self.baseline_metrics.items():
            if len(data_points) < 10:  # Need enough data
                continue
            
            try:
                # Simple statistical anomaly detection
                values = [dp["value"] for dp in data_points[-50:]]  # Last 50 points
                
                if len(values) < 10:
                    continue
                
                mean_val = statistics.mean(values)
                stdev_val = statistics.stdev(values) if len(values) > 1 else 0
                
                # Check if latest value is anomalous
                latest_value = values[-1]
                z_score = abs(latest_value - mean_val) / stdev_val if stdev_val > 0 else 0
                
                if z_score > 3:  # 3 sigma rule
                    await self._handle_anomaly(metric_key, latest_value, mean_val, z_score)
                    
            except Exception as e:
                logger.error(f"Error detecting anomalies for {metric_key}: {e}")
    
    async def _handle_anomaly(self, metric_key: str, value: float, baseline: float, z_score: float):
        """Handle detected anomaly"""
        logger.warning(f"ANOMALY DETECTED: {metric_key} = {value:.2f} (baseline: {baseline:.2f}, z-score: {z_score:.2f})")
        
        # In real implementation, create dynamic alerts or notifications
    
    # Public API methods
    
    def record_trace_span(self, span: TraceSpan):
        """Record a distributed trace span"""
        if span.trace_id not in self.active_traces:
            self.active_traces[span.trace_id] = []
        
        self.active_traces[span.trace_id].append(span)
        
        # Move completed traces to history
        if span.status in [TraceStatus.OK, TraceStatus.ERROR, TraceStatus.CANCELLED]:
            # Check if this is the root span or all spans are complete
            trace_spans = self.active_traces[span.trace_id]
            if len(trace_spans) > 0:  # Simple completion check
                self.trace_history.extend(trace_spans)
                del self.active_traces[span.trace_id]
    
    def record_log(self, log_entry: LogEntry):
        """Record a log entry"""
        self.log_buffer.append(log_entry)
        
        # Keep buffer size manageable
        if len(self.log_buffer) > 10000:
            self.log_buffer = self.log_buffer[-5000:]  # Keep last 5000
    
    def record_rum_metric(self, rum_metric: RUMMetric):
        """Record a Real User Monitoring metric"""
        self.rum_metrics.append(rum_metric)
        
        # Update session data
        self.rum_sessions[rum_metric.session_id] = rum_metric
        
        # Keep only recent RUM data
        cutoff = datetime.now() - timedelta(hours=24)
        self.rum_metrics = [m for m in self.rum_metrics if m.timestamp > cutoff]
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """List all dashboards"""
        return list(self.dashboards.values())
    
    def get_alerts(self, status: Optional[AlertStatus] = None) -> List[Alert]:
        """Get alerts, optionally filtered by status"""
        alerts = list(self.alerts.values())
        if status:
            alerts = [a for a in alerts if a.status == status]
        return alerts
    
    def get_slo_status(self, slo_name: Optional[str] = None) -> Union[SLO, List[SLO]]:
        """Get SLO status"""
        if slo_name:
            return self.slos.get(slo_name)
        return list(self.slos.values())
    
    def get_trace_history(self, trace_id: Optional[str] = None, limit: int = 100) -> List[TraceSpan]:
        """Get trace history"""
        if trace_id:
            return [span for span in self.trace_history if span.trace_id == trace_id]
        return self.trace_history[-limit:]
    
    def get_rum_analytics(self, time_range: str = "1h") -> Dict[str, Any]:
        """Get RUM analytics"""
        # Parse time range
        if time_range == "1h":
            cutoff = datetime.now() - timedelta(hours=1)
        elif time_range == "24h":
            cutoff = datetime.now() - timedelta(hours=24)
        elif time_range == "7d":
            cutoff = datetime.now() - timedelta(days=7)
        else:
            cutoff = datetime.now() - timedelta(hours=1)
        
        # Filter metrics
        recent_metrics = [m for m in self.rum_metrics if m.timestamp > cutoff]
        
        if not recent_metrics:
            return {"error": "No RUM data available"}
        
        # Calculate analytics
        page_load_times = [m.page_load_time_ms for m in recent_metrics]
        fcp_times = [m.first_contentful_paint_ms for m in recent_metrics]
        lcp_times = [m.largest_contentful_paint_ms for m in recent_metrics]
        
        analytics = {
            "total_sessions": len(set(m.session_id for m in recent_metrics)),
            "total_page_views": len(recent_metrics),
            "performance": {
                "avg_page_load_time_ms": statistics.mean(page_load_times),
                "p95_page_load_time_ms": sorted(page_load_times)[int(len(page_load_times) * 0.95)] if page_load_times else 0,
                "avg_first_contentful_paint_ms": statistics.mean(fcp_times),
                "avg_largest_contentful_paint_ms": statistics.mean(lcp_times)
            },
            "errors": {
                "total_errors": sum(len(m.errors) for m in recent_metrics),
                "error_rate": sum(len(m.errors) for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
            },
            "geographic_distribution": self._calculate_geographic_distribution(recent_metrics),
            "time_range": time_range,
            "generated_at": datetime.now().isoformat()
        }
        
        return analytics
    
    def _calculate_geographic_distribution(self, metrics: List[RUMMetric]) -> Dict[str, int]:
        """Calculate geographic distribution of users"""
        distribution = {}
        for metric in metrics:
            country = metric.country
            distribution[country] = distribution.get(country, 0) + 1
        return distribution
    
    def generate_observability_report(self) -> Dict[str, Any]:
        """Generate comprehensive observability report"""
        # Get current status
        firing_alerts = self.get_alerts(AlertStatus.FIRING)
        slo_status = self.get_slo_status()
        rum_analytics = self.get_rum_analytics("24h")
        
        # Calculate health score
        health_score = self._calculate_health_score()
        
        report = {
            "summary": {
                "health_score": health_score,
                "total_alerts": len(self.alerts),
                "firing_alerts": len(firing_alerts),
                "total_slos": len(self.slos),
                "healthy_slos": len([slo for slo in slo_status if slo.status == SLOStatus.HEALTHY]),
                "total_dashboards": len(self.dashboards),
                "active_traces": len(self.active_traces),
                "rum_sessions_24h": rum_analytics.get("total_sessions", 0)
            },
            "alerts": {
                "firing": [{
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "fired_at": alert.fired_at.isoformat() if alert.fired_at else None
                } for alert in firing_alerts],
                "summary_by_severity": self._summarize_alerts_by_severity()
            },
            "slos": [{
                "name": slo.name,
                "status": slo.status.value,
                "current_percentage": slo.current_percentage,
                "target_percentage": slo.target_percentage,
                "error_budget_remaining": slo.error_budget_remaining
            } for slo in slo_status],
            "performance": rum_analytics.get("performance", {}),
            "dashboards": [{
                "id": dashboard.id,
                "title": dashboard.title,
                "tags": dashboard.tags
            } for dashboard in self.dashboards.values()],
            "recommendations": self._generate_observability_recommendations(),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        # Deduct for firing alerts
        firing_alerts = self.get_alerts(AlertStatus.FIRING)
        for alert in firing_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                score -= 20
            elif alert.severity == AlertSeverity.HIGH:
                score -= 10
            elif alert.severity == AlertSeverity.MEDIUM:
                score -= 5
            else:
                score -= 1
        
        # Deduct for SLO breaches
        slo_status = self.get_slo_status()
        for slo in slo_status:
            if slo.status == SLOStatus.BREACHED:
                score -= 15
            elif slo.status == SLOStatus.CRITICAL:
                score -= 10
            elif slo.status == SLOStatus.WARNING:
                score -= 5
        
        return max(0.0, score)
    
    def _summarize_alerts_by_severity(self) -> Dict[str, int]:
        """Summarize alerts by severity"""
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.alerts.values():
            if alert.status == AlertStatus.FIRING:
                summary[alert.severity.value] += 1
        
        return summary
    
    def _generate_observability_recommendations(self) -> List[str]:
        """Generate observability recommendations"""
        recommendations = []
        
        # Check for missing dashboards
        if len(self.dashboards) < 4:
            recommendations.append("Consider creating additional dashboards for better visibility")
        
        # Check for SLO coverage
        if len(self.slos) < 3:
            recommendations.append("Define more SLOs to improve service reliability tracking")
        
        # Check alert coverage
        if len(self.alerts) < 5:
            recommendations.append("Add more alert rules to catch potential issues early")
        
        # Check for high error budget consumption
        slo_status = self.get_slo_status()
        for slo in slo_status:
            if slo.error_budget_remaining < 20:
                recommendations.append(f"SLO '{slo.name}' has low error budget remaining ({slo.error_budget_remaining:.1f}%)")
        
        # General recommendations
        recommendations.extend([
            "Regularly review and update alert thresholds",
            "Implement automated runbooks for common alerts",
            "Set up synthetic monitoring for critical user journeys",
            "Consider implementing chaos engineering to test observability"
        ])
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_full_stack_observer():
        """Test the Full-Stack Observer"""
        observer = FullStackObserver()
        
        # Initialize
        await observer.initialize()
        
        # Simulate some activity
        print("Simulating observability data...")
        
        # Record some trace spans
        trace_span = TraceSpan(
            trace_id="trace-001",
            span_id="span-001",
            parent_span_id=None,
            operation_name="http_request",
            service_name="api",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=150),
            duration_ms=150.0,
            status=TraceStatus.OK,
            tags={"http.method": "GET", "http.url": "/api/news"},
            logs=[],
            error_message=None
        )
        observer.record_trace_span(trace_span)
        
        # Record some logs
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            service="api",
            component="news_handler",
            message="Successfully processed news request",
            trace_id="trace-001",
            span_id="span-001",
            labels={"environment": "prod"},
            fields={"response_time_ms": 150},
            source="application"
        )
        observer.record_log(log_entry)
        
        # Record RUM metrics
        rum_metric = RUMMetric(
            session_id="session-001",
            user_id="user-123",
            page_url="https://example.com/news",
            user_agent="Mozilla/5.0...",
            country="US",
            city="San Francisco",
            connection_type="4g",
            page_load_time_ms=2500.0,
            first_contentful_paint_ms=1200.0,
            largest_contentful_paint_ms=2000.0,
            cumulative_layout_shift=0.05,
            first_input_delay_ms=50.0,
            time_to_interactive_ms=2200.0,
            errors=[],
            custom_metrics={"api_calls": 3},
            timestamp=datetime.now()
        )
        observer.record_rum_metric(rum_metric)
        
        # Wait for background processing
        await asyncio.sleep(5)
        
        # Get observability report
        report = observer.generate_observability_report()
        print(f"Observability Report: {json.dumps(report, indent=2, default=str)}")
        
        # Get RUM analytics
        rum_analytics = observer.get_rum_analytics("1h")
        print(f"RUM Analytics: {json.dumps(rum_analytics, indent=2, default=str)}")
        
        # List dashboards
        dashboards = observer.list_dashboards()
        print(f"Available Dashboards: {[d.title for d in dashboards]}")
        
        # Get SLO status
        slo_status = observer.get_slo_status()
        print(f"SLO Status: {[(slo.name, slo.status.value, slo.current_percentage) for slo in slo_status]}")
    
    # Run test
    asyncio.run(test_full_stack_observer())