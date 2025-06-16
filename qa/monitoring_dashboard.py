#!/usr/bin/env python3
"""
Dr. Orion "TestMaster" Vanguard - Superhuman QA Monitoring Dashboard

Real-time monitoring and alerting system for:
- Test execution metrics
- AI inference performance
- Chaos experiment results
- Cache performance analytics
- Multi-agent pipeline status
- Predictive failure analysis

Author: Dr. Orion "TestMaster" Vanguard
Version: 1.0.0
License: MIT
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import threading
import queue
import sqlite3
import yaml
import requests
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis
import psutil

@dataclass
class TestMetric:
    """Test execution metric"""
    timestamp: datetime
    test_type: str
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    environment: str
    persona: Optional[str] = None
    model: Optional[str] = None
    experiment: Optional[str] = None
    agent: Optional[str] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None

@dataclass
class SystemMetric:
    """System performance metric"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    cache_hit_rate: float
    api_response_time: float
    error_rate: float

@dataclass
class Alert:
    """Alert definition"""
    id: str
    severity: str  # critical, warning, info
    title: str
    message: str
    timestamp: datetime
    test_type: Optional[str] = None
    environment: Optional[str] = None
    acknowledged: bool = False
    resolved: bool = False

class SuperhumanQAMonitor:
    """Advanced monitoring system for superhuman QA"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.db_path = "qa_monitoring.db"
        self.redis_client = self._init_redis()
        
        # Metrics storage
        self.test_metrics = deque(maxlen=10000)
        self.system_metrics = deque(maxlen=1000)
        self.alerts = deque(maxlen=500)
        
        # Real-time data
        self.active_tests = {}
        self.test_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        
        # ML models for anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_model_trained = False
        
        # Initialize database
        self._init_database()
        
        # Start background monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._background_monitoring)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Flask app for dashboard
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'superhuman-qa-monitor'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._setup_routes()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('monitoring', {})
        except Exception as e:
            logging.warning(f"Could not load config: {e}")
            return self._get_default_monitoring_config()
    
    def _get_default_monitoring_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration"""
        return {
            "thresholds": {
                "test_failure_rate": 0.1,
                "response_time": 5.0,
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "cache_hit_rate": 0.8,
                "error_rate": 0.05
            },
            "alerts": {
                "slack_webhook": None,
                "email_smtp": None,
                "pagerduty_key": None
            },
            "retention": {
                "metrics_days": 30,
                "alerts_days": 7
            }
        }
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection for caching metrics"""
        try:
            client = redis.Redis(
                host=self.config.get('redis', {}).get('host', 'localhost'),
                port=self.config.get('redis', {}).get('port', 6379),
                decode_responses=True
            )
            client.ping()
            return client
        except Exception as e:
            logging.warning(f"Redis not available: {e}")
            return None
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                test_type TEXT NOT NULL,
                test_name TEXT NOT NULL,
                status TEXT NOT NULL,
                duration REAL NOT NULL,
                environment TEXT NOT NULL,
                persona TEXT,
                model TEXT,
                experiment TEXT,
                agent TEXT,
                error_message TEXT,
                confidence_score REAL,
                resource_usage TEXT
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                disk_usage REAL NOT NULL,
                network_io TEXT NOT NULL,
                active_connections INTEGER NOT NULL,
                cache_hit_rate REAL NOT NULL,
                api_response_time REAL NOT NULL,
                error_rate REAL NOT NULL
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                test_type TEXT,
                environment TEXT,
                acknowledged INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_test_metric(self, metric: TestMetric):
        """Record a test execution metric"""
        self.test_metrics.append(metric)
        self.test_queue.put(metric)
        
        # Store in database
        self._store_test_metric(metric)
        
        # Cache in Redis
        if self.redis_client:
            key = f"test_metric:{metric.timestamp.isoformat()}"
            self.redis_client.setex(key, 3600, json.dumps(asdict(metric), default=str))
        
        # Check for alerts
        self._check_test_alerts(metric)
        
        # Emit real-time update
        self.socketio.emit('test_metric', asdict(metric), namespace='/monitor')
    
    def record_system_metric(self, metric: SystemMetric):
        """Record a system performance metric"""
        self.system_metrics.append(metric)
        
        # Store in database
        self._store_system_metric(metric)
        
        # Check for alerts
        self._check_system_alerts(metric)
        
        # Emit real-time update
        self.socketio.emit('system_metric', asdict(metric), namespace='/monitor')
    
    def _store_test_metric(self, metric: TestMetric):
        """Store test metric in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_metrics (
                timestamp, test_type, test_name, status, duration, environment,
                persona, model, experiment, agent, error_message, confidence_score, resource_usage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.timestamp.isoformat(),
            metric.test_type,
            metric.test_name,
            metric.status,
            metric.duration,
            metric.environment,
            metric.persona,
            metric.model,
            metric.experiment,
            metric.agent,
            metric.error_message,
            metric.confidence_score,
            json.dumps(metric.resource_usage) if metric.resource_usage else None
        ))
        
        conn.commit()
        conn.close()
    
    def _store_system_metric(self, metric: SystemMetric):
        """Store system metric in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_metrics (
                timestamp, cpu_percent, memory_percent, disk_usage, network_io,
                active_connections, cache_hit_rate, api_response_time, error_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.timestamp.isoformat(),
            metric.cpu_percent,
            metric.memory_percent,
            metric.disk_usage,
            json.dumps(metric.network_io),
            metric.active_connections,
            metric.cache_hit_rate,
            metric.api_response_time,
            metric.error_rate
        ))
        
        conn.commit()
        conn.close()
    
    def _check_test_alerts(self, metric: TestMetric):
        """Check for test-related alerts"""
        # Test failure rate alert
        recent_tests = [m for m in self.test_metrics 
                       if m.timestamp > datetime.now() - timedelta(minutes=30)
                       and m.test_type == metric.test_type]
        
        if len(recent_tests) >= 5:
            failure_rate = len([t for t in recent_tests if t.status == 'failed']) / len(recent_tests)
            threshold = self.config.get('thresholds', {}).get('test_failure_rate', 0.1)
            
            if failure_rate > threshold:
                alert = Alert(
                    id=f"test_failure_{metric.test_type}_{int(time.time())}",
                    severity="critical",
                    title=f"High Failure Rate: {metric.test_type}",
                    message=f"Failure rate {failure_rate:.2%} exceeds threshold {threshold:.2%}",
                    timestamp=datetime.now(),
                    test_type=metric.test_type,
                    environment=metric.environment
                )
                self._create_alert(alert)
        
        # Long duration alert
        if metric.duration > 300:  # 5 minutes
            alert = Alert(
                id=f"long_duration_{metric.test_name}_{int(time.time())}",
                severity="warning",
                title=f"Long Test Duration: {metric.test_name}",
                message=f"Test took {metric.duration:.1f}s to complete",
                timestamp=datetime.now(),
                test_type=metric.test_type,
                environment=metric.environment
            )
            self._create_alert(alert)
        
        # Low confidence score alert
        if metric.confidence_score and metric.confidence_score < 0.7:
            alert = Alert(
                id=f"low_confidence_{metric.test_name}_{int(time.time())}",
                severity="warning",
                title=f"Low Confidence Score: {metric.test_name}",
                message=f"Confidence score {metric.confidence_score:.2f} below threshold",
                timestamp=datetime.now(),
                test_type=metric.test_type,
                environment=metric.environment
            )
            self._create_alert(alert)
    
    def _check_system_alerts(self, metric: SystemMetric):
        """Check for system-related alerts"""
        thresholds = self.config.get('thresholds', {})
        
        # CPU usage alert
        if metric.cpu_percent > thresholds.get('cpu_usage', 80):
            alert = Alert(
                id=f"high_cpu_{int(time.time())}",
                severity="warning",
                title="High CPU Usage",
                message=f"CPU usage {metric.cpu_percent:.1f}% exceeds threshold",
                timestamp=datetime.now()
            )
            self._create_alert(alert)
        
        # Memory usage alert
        if metric.memory_percent > thresholds.get('memory_usage', 85):
            alert = Alert(
                id=f"high_memory_{int(time.time())}",
                severity="critical",
                title="High Memory Usage",
                message=f"Memory usage {metric.memory_percent:.1f}% exceeds threshold",
                timestamp=datetime.now()
            )
            self._create_alert(alert)
        
        # Cache hit rate alert
        if metric.cache_hit_rate < thresholds.get('cache_hit_rate', 0.8):
            alert = Alert(
                id=f"low_cache_hit_{int(time.time())}",
                severity="warning",
                title="Low Cache Hit Rate",
                message=f"Cache hit rate {metric.cache_hit_rate:.2%} below threshold",
                timestamp=datetime.now()
            )
            self._create_alert(alert)
        
        # API response time alert
        if metric.api_response_time > thresholds.get('response_time', 5.0):
            alert = Alert(
                id=f"slow_api_{int(time.time())}",
                severity="warning",
                title="Slow API Response",
                message=f"API response time {metric.api_response_time:.2f}s exceeds threshold",
                timestamp=datetime.now()
            )
            self._create_alert(alert)
    
    def _create_alert(self, alert: Alert):
        """Create and process an alert"""
        self.alerts.append(alert)
        self.alert_queue.put(alert)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alerts (
                id, severity, title, message, timestamp, test_type, environment
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id,
            alert.severity,
            alert.title,
            alert.message,
            alert.timestamp.isoformat(),
            alert.test_type,
            alert.environment
        ))
        
        conn.commit()
        conn.close()
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        # Emit real-time update
        self.socketio.emit('alert', asdict(alert), namespace='/monitor')
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications via configured channels"""
        # Slack notification
        slack_webhook = self.config.get('alerts', {}).get('slack_webhook')
        if slack_webhook and alert.severity in ['critical', 'warning']:
            self._send_slack_notification(alert, slack_webhook)
        
        # Email notification (implementation would depend on SMTP config)
        # PagerDuty notification (implementation would depend on PD config)
    
    def _send_slack_notification(self, alert: Alert, webhook_url: str):
        """Send Slack notification"""
        try:
            color = {
                'critical': '#FF0000',
                'warning': '#FFA500',
                'info': '#0000FF'
            }.get(alert.severity, '#808080')
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"🚨 {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ]
                }]
            }
            
            if alert.test_type:
                payload["attachments"][0]["fields"].append(
                    {"title": "Test Type", "value": alert.test_type, "short": True}
                )
            
            if alert.environment:
                payload["attachments"][0]["fields"].append(
                    {"title": "Environment", "value": alert.environment, "short": True}
                )
            
            requests.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")
    
    def _background_monitoring(self):
        """Background thread for system monitoring"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metric = self._collect_system_metrics()
                if metric:
                    self.record_system_metric(metric)
                
                # Train anomaly detection model
                if len(self.system_metrics) > 100 and not self.is_model_trained:
                    self._train_anomaly_detector()
                
                # Detect anomalies
                if self.is_model_trained and len(self.system_metrics) > 0:
                    self._detect_anomalies()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                time.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logging.error(f"Background monitoring error: {e}")
                time.sleep(60)
    
    def _collect_system_metrics(self) -> Optional[SystemMetric]:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Application metrics (mock implementation)
            cache_hit_rate = self._get_cache_hit_rate()
            api_response_time = self._get_api_response_time()
            error_rate = self._get_error_rate()
            active_connections = self._get_active_connections()
            
            return SystemMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                active_connections=active_connections,
                cache_hit_rate=cache_hit_rate,
                api_response_time=api_response_time,
                error_rate=error_rate
            )
        except Exception as e:
            logging.error(f"Failed to collect system metrics: {e}")
            return None
    
    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate (mock implementation)"""
        if self.redis_client:
            try:
                info = self.redis_client.info()
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                total = hits + misses
                return hits / total if total > 0 else 0.0
            except:
                pass
        return 0.85  # Mock value
    
    def _get_api_response_time(self) -> float:
        """Get average API response time (mock implementation)"""
        # In a real implementation, this would query application metrics
        return 1.2  # Mock value
    
    def _get_error_rate(self) -> float:
        """Get application error rate (mock implementation)"""
        # In a real implementation, this would query application logs
        return 0.02  # Mock value
    
    def _get_active_connections(self) -> int:
        """Get number of active connections (mock implementation)"""
        return len(psutil.net_connections())
    
    def _train_anomaly_detector(self):
        """Train anomaly detection model"""
        try:
            # Prepare training data
            data = []
            for metric in list(self.system_metrics)[-100:]:
                data.append([
                    metric.cpu_percent,
                    metric.memory_percent,
                    metric.disk_usage,
                    metric.cache_hit_rate,
                    metric.api_response_time,
                    metric.error_rate
                ])
            
            if len(data) >= 50:
                X = np.array(data)
                X_scaled = self.scaler.fit_transform(X)
                self.anomaly_detector.fit(X_scaled)
                self.is_model_trained = True
                logging.info("Anomaly detection model trained successfully")
        except Exception as e:
            logging.error(f"Failed to train anomaly detector: {e}")
    
    def _detect_anomalies(self):
        """Detect system anomalies"""
        try:
            if len(self.system_metrics) == 0:
                return
            
            latest_metric = self.system_metrics[-1]
            data = np.array([[
                latest_metric.cpu_percent,
                latest_metric.memory_percent,
                latest_metric.disk_usage,
                latest_metric.cache_hit_rate,
                latest_metric.api_response_time,
                latest_metric.error_rate
            ]])
            
            data_scaled = self.scaler.transform(data)
            anomaly_score = self.anomaly_detector.decision_function(data_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(data_scaled)[0] == -1
            
            if is_anomaly:
                alert = Alert(
                    id=f"anomaly_{int(time.time())}",
                    severity="warning",
                    title="System Anomaly Detected",
                    message=f"Anomaly score: {anomaly_score:.3f}",
                    timestamp=datetime.now()
                )
                self._create_alert(alert)
        except Exception as e:
            logging.error(f"Failed to detect anomalies: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            retention_days = self.config.get('retention', {}).get('metrics_days', 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean up old test metrics
            cursor.execute(
                "DELETE FROM test_metrics WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            # Clean up old system metrics
            cursor.execute(
                "DELETE FROM system_metrics WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            # Clean up old alerts
            alert_retention_days = self.config.get('retention', {}).get('alerts_days', 7)
            alert_cutoff_date = datetime.now() - timedelta(days=alert_retention_days)
            cursor.execute(
                "DELETE FROM alerts WHERE timestamp < ?",
                (alert_cutoff_date.isoformat(),)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes for dashboard"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('monitoring_dashboard.html')
        
        @self.app.route('/api/metrics/test')
        def get_test_metrics():
            hours = request.args.get('hours', 24, type=int)
            cutoff = datetime.now() - timedelta(hours=hours)
            
            recent_metrics = [
                asdict(m) for m in self.test_metrics 
                if m.timestamp > cutoff
            ]
            
            return jsonify(recent_metrics)
        
        @self.app.route('/api/metrics/system')
        def get_system_metrics():
            hours = request.args.get('hours', 24, type=int)
            cutoff = datetime.now() - timedelta(hours=hours)
            
            recent_metrics = [
                asdict(m) for m in self.system_metrics 
                if m.timestamp > cutoff
            ]
            
            return jsonify(recent_metrics)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            active_alerts = [
                asdict(a) for a in self.alerts 
                if not a.resolved
            ]
            
            return jsonify(active_alerts)
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    # Update in database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                        (alert_id,)
                    )
                    conn.commit()
                    conn.close()
                    return jsonify({'status': 'acknowledged'})
            
            return jsonify({'error': 'Alert not found'}), 404
        
        @self.app.route('/api/dashboard/summary')
        def get_dashboard_summary():
            # Calculate summary statistics
            recent_tests = [
                m for m in self.test_metrics 
                if m.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            total_tests = len(recent_tests)
            passed_tests = len([t for t in recent_tests if t.status == 'passed'])
            failed_tests = len([t for t in recent_tests if t.status == 'failed'])
            
            active_alerts_count = len([a for a in self.alerts if not a.resolved])
            critical_alerts = len([a for a in self.alerts if not a.resolved and a.severity == 'critical'])
            
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            
            return jsonify({
                'total_tests_24h': total_tests,
                'passed_tests_24h': passed_tests,
                'failed_tests_24h': failed_tests,
                'success_rate_24h': passed_tests / total_tests if total_tests > 0 else 0,
                'active_alerts': active_alerts_count,
                'critical_alerts': critical_alerts,
                'current_cpu': latest_system.cpu_percent if latest_system else 0,
                'current_memory': latest_system.memory_percent if latest_system else 0,
                'cache_hit_rate': latest_system.cache_hit_rate if latest_system else 0,
                'api_response_time': latest_system.api_response_time if latest_system else 0
            })
        
        @self.socketio.on('connect', namespace='/monitor')
        def handle_connect():
            emit('connected', {'status': 'Connected to monitoring dashboard'})
    
    def run_dashboard(self, host='0.0.0.0', port=5000, debug=False):
        """Run the monitoring dashboard"""
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

def main():
    """Main function for standalone monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dr. Orion TestMaster - Superhuman QA Monitoring Dashboard"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Dashboard host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Dashboard port"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize monitor
    monitor = SuperhumanQAMonitor(args.config)
    
    print(f"Starting Superhuman QA Monitoring Dashboard on {args.host}:{args.port}")
    print("Dashboard will be available at: http://localhost:5000")
    
    try:
        monitor.run_dashboard(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down monitoring dashboard...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()