#!/usr/bin/env python3
"""
Commander Solaris "DeployX" Vivante - Web Dashboard Interface
Superhuman Deployment Strategist & Resilience Commander

This module provides a comprehensive web-based dashboard for DeployX operations,
built with Flask and modern web technologies. Features real-time monitoring,
interactive deployment management, chaos engineering control, and AI-enhanced
canary analysis visualization.

Features:
- Real-time deployment monitoring
- Interactive canary analysis dashboard
- Multi-region deployment visualization
- Chaos engineering experiment control
- Security and compliance dashboards
- AI-powered insights and recommendations
- WebSocket-based real-time updates
- Responsive design with modern UI

Author: Commander Solaris "DeployX" Vivante
Version: 1.0.0
Date: 2023-12-01
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_socketio import SocketIO, emit, join_room, leave_room
    from flask_cors import CORS
except ImportError as e:
    print(f"Error: Flask dependencies not installed: {e}")
    print("Install with: pip install flask flask-socketio flask-cors")
    exit(1)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
except ImportError:
    print("Warning: Plotly not installed. Charts will be disabled.")
    plotly = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeployX-Dashboard')

# Flask app configuration
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.config['SECRET_KEY'] = 'deployX-superhuman-deployment-strategist'
app.config['DEBUG'] = True

# Enable CORS and SocketIO
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"

class MetricStatus(Enum):
    """Metric status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class DeploymentMetrics:
    """Deployment metrics data structure"""
    deployment_id: str
    application: str
    version: str
    environment: str
    status: DeploymentStatus
    progress: float
    success_rate: float
    error_rate: float
    latency_p95: float
    throughput: float
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    canary_confidence: Optional[float] = None
    traffic_split: Optional[int] = None

@dataclass
class RegionStatus:
    """Region status data structure"""
    region: str
    status: str
    health: MetricStatus
    deployments: int
    latency: float
    availability: float
    last_update: datetime

@dataclass
class ChaosExperiment:
    """Chaos experiment data structure"""
    experiment_id: str
    name: str
    type: str
    target: str
    status: str
    start_time: datetime
    duration: int
    intensity: float
    resilience_score: Optional[float] = None
    impact_metrics: Optional[Dict[str, float]] = None

class DeployXDashboard:
    """Main dashboard class for Commander DeployX"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.active_deployments = {}
        self.region_status = {}
        self.chaos_experiments = {}
        self.metrics_history = []
        self.alerts = []
        
        # Initialize mock data
        self._initialize_mock_data()
        
        logger.info("DeployX Dashboard initialized")
    
    def _initialize_mock_data(self):
        """Initialize mock data for demonstration"""
        # Mock deployments
        self.active_deployments = {
            'deploy-001': DeploymentMetrics(
                deployment_id='deploy-001',
                application='ai-news-dashboard',
                version='2.1.0',
                environment='production',
                status=DeploymentStatus.RUNNING,
                progress=75.0,
                success_rate=99.2,
                error_rate=0.008,
                latency_p95=145.0,
                throughput=2500.0,
                start_time=datetime.now() - timedelta(minutes=15),
                estimated_completion=datetime.now() + timedelta(minutes=5),
                canary_confidence=94.5,
                traffic_split=25
            ),
            'deploy-002': DeploymentMetrics(
                deployment_id='deploy-002',
                application='user-service',
                version='1.8.2',
                environment='staging',
                status=DeploymentStatus.SUCCESS,
                progress=100.0,
                success_rate=98.8,
                error_rate=0.012,
                latency_p95=89.0,
                throughput=1800.0,
                start_time=datetime.now() - timedelta(hours=2),
                canary_confidence=96.2,
                traffic_split=100
            )
        }
        
        # Mock region status
        self.region_status = {
            'us-east-1': RegionStatus(
                region='us-east-1',
                status='active',
                health=MetricStatus.HEALTHY,
                deployments=12,
                latency=45.2,
                availability=99.98,
                last_update=datetime.now()
            ),
            'eu-west-1': RegionStatus(
                region='eu-west-1',
                status='active',
                health=MetricStatus.WARNING,
                deployments=8,
                latency=78.5,
                availability=99.85,
                last_update=datetime.now() - timedelta(minutes=2)
            ),
            'ap-southeast-1': RegionStatus(
                region='ap-southeast-1',
                status='maintenance',
                health=MetricStatus.UNKNOWN,
                deployments=3,
                latency=125.0,
                availability=99.50,
                last_update=datetime.now() - timedelta(minutes=10)
            )
        }
        
        # Mock chaos experiments
        self.chaos_experiments = {
            'chaos-001': ChaosExperiment(
                experiment_id='chaos-001',
                name='Pod Failure Resilience',
                type='pod-kill',
                target='ai-news-dashboard',
                status='running',
                start_time=datetime.now() - timedelta(minutes=5),
                duration=300,
                intensity=0.3,
                resilience_score=92.3,
                impact_metrics={
                    'availability': 99.95,
                    'response_time': 1.15,
                    'error_rate': 0.002
                }
            )
        }
        
        # Mock alerts
        self.alerts = [
            {
                'id': 'alert-001',
                'severity': 'warning',
                'title': 'High Latency in EU Region',
                'message': 'P95 latency increased by 15% in eu-west-1',
                'timestamp': datetime.now() - timedelta(minutes=5),
                'status': 'active'
            },
            {
                'id': 'alert-002',
                'severity': 'info',
                'title': 'Canary Ready for Promotion',
                'message': 'ai-news-dashboard v2.1.0 canary analysis complete',
                'timestamp': datetime.now() - timedelta(minutes=2),
                'status': 'active'
            }
        ]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            'deployments': {k: asdict(v) for k, v in self.active_deployments.items()},
            'regions': {k: asdict(v) for k, v in self.region_status.items()},
            'chaos_experiments': {k: asdict(v) for k, v in self.chaos_experiments.items()},
            'alerts': self.alerts,
            'summary': self._get_summary_metrics(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics"""
        total_deployments = len(self.active_deployments)
        running_deployments = sum(1 for d in self.active_deployments.values() 
                                if d.status == DeploymentStatus.RUNNING)
        
        avg_success_rate = sum(d.success_rate for d in self.active_deployments.values()) / max(total_deployments, 1)
        avg_latency = sum(d.latency_p95 for d in self.active_deployments.values()) / max(total_deployments, 1)
        
        healthy_regions = sum(1 for r in self.region_status.values() 
                            if r.health == MetricStatus.HEALTHY)
        
        active_experiments = sum(1 for e in self.chaos_experiments.values() 
                               if e.status == 'running')
        
        return {
            'total_deployments': total_deployments,
            'running_deployments': running_deployments,
            'avg_success_rate': round(avg_success_rate, 2),
            'avg_latency': round(avg_latency, 1),
            'healthy_regions': healthy_regions,
            'total_regions': len(self.region_status),
            'active_experiments': active_experiments,
            'active_alerts': len([a for a in self.alerts if a['status'] == 'active'])
        }
    
    def create_deployment_timeline_chart(self) -> str:
        """Create deployment timeline chart"""
        if not plotly:
            return json.dumps({})
        
        # Mock timeline data
        timeline_data = []
        for i, (dep_id, deployment) in enumerate(self.active_deployments.items()):
            timeline_data.append({
                'Task': deployment.application,
                'Start': deployment.start_time,
                'Finish': deployment.estimated_completion or datetime.now(),
                'Resource': deployment.environment,
                'Status': deployment.status.value
            })
        
        if not timeline_data:
            return json.dumps({})
        
        fig = px.timeline(
            timeline_data,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Status",
            title="Deployment Timeline"
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Application"
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_metrics_chart(self) -> str:
        """Create metrics visualization chart"""
        if not plotly:
            return json.dumps({})
        
        # Mock metrics data
        timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(12, 0, -1)]
        
        fig = go.Figure()
        
        # Success rate
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[99.2, 99.1, 99.3, 99.0, 98.8, 99.1, 99.2, 99.4, 99.1, 99.2, 99.3, 99.2],
            mode='lines+markers',
            name='Success Rate (%)',
            line=dict(color='green')
        ))
        
        # Latency
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[145, 142, 148, 150, 155, 149, 145, 143, 147, 145, 144, 145],
            mode='lines+markers',
            name='Latency P95 (ms)',
            yaxis='y2',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='System Metrics Over Time',
            xaxis_title='Time',
            yaxis=dict(
                title='Success Rate (%)',
                side='left'
            ),
            yaxis2=dict(
                title='Latency (ms)',
                side='right',
                overlaying='y'
            ),
            height=400
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_region_map_chart(self) -> str:
        """Create world map showing region status"""
        if not plotly:
            return json.dumps({})
        
        # Mock geographic data
        region_coords = {
            'us-east-1': {'lat': 39.0458, 'lon': -76.6413, 'name': 'US East (Virginia)'},
            'eu-west-1': {'lat': 53.4084, 'lon': -8.2439, 'name': 'EU West (Ireland)'},
            'ap-southeast-1': {'lat': 1.3521, 'lon': 103.8198, 'name': 'Asia Pacific (Singapore)'}
        }
        
        lats, lons, texts, colors = [], [], [], []
        
        for region_id, region in self.region_status.items():
            if region_id in region_coords:
                coord = region_coords[region_id]
                lats.append(coord['lat'])
                lons.append(coord['lon'])
                texts.append(f"{coord['name']}<br>Status: {region.status}<br>Health: {region.health.value}<br>Availability: {region.availability}%")
                
                if region.health == MetricStatus.HEALTHY:
                    colors.append('green')
                elif region.health == MetricStatus.WARNING:
                    colors.append('orange')
                elif region.health == MetricStatus.CRITICAL:
                    colors.append('red')
                else:
                    colors.append('gray')
        
        fig = go.Figure(data=go.Scattergeo(
            lon=lons,
            lat=lats,
            text=texts,
            mode='markers',
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color='white')
            )
        ))
        
        fig.update_layout(
            title='Global Region Status',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=400
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)

# Initialize dashboard
dashboard = DeployXDashboard()

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/dashboard')
def api_dashboard():
    """Get dashboard data API"""
    return jsonify(dashboard.get_dashboard_data())

@app.route('/api/deployments')
def api_deployments():
    """Get deployments API"""
    deployments = {k: asdict(v) for k, v in dashboard.active_deployments.items()}
    return jsonify(deployments)

@app.route('/api/deployments/<deployment_id>')
def api_deployment_detail(deployment_id):
    """Get specific deployment details"""
    if deployment_id in dashboard.active_deployments:
        return jsonify(asdict(dashboard.active_deployments[deployment_id]))
    return jsonify({'error': 'Deployment not found'}), 404

@app.route('/api/deployments/<deployment_id>/promote', methods=['POST'])
def api_promote_canary(deployment_id):
    """Promote canary deployment"""
    if deployment_id in dashboard.active_deployments:
        deployment = dashboard.active_deployments[deployment_id]
        deployment.traffic_split = 100
        deployment.status = DeploymentStatus.SUCCESS
        
        # Emit real-time update
        socketio.emit('deployment_updated', {
            'deployment_id': deployment_id,
            'data': asdict(deployment)
        })
        
        return jsonify({'status': 'promoted', 'deployment_id': deployment_id})
    return jsonify({'error': 'Deployment not found'}), 404

@app.route('/api/deployments/<deployment_id>/rollback', methods=['POST'])
def api_rollback_deployment(deployment_id):
    """Rollback deployment"""
    if deployment_id in dashboard.active_deployments:
        deployment = dashboard.active_deployments[deployment_id]
        deployment.status = DeploymentStatus.ROLLED_BACK
        deployment.progress = 0.0
        
        # Emit real-time update
        socketio.emit('deployment_updated', {
            'deployment_id': deployment_id,
            'data': asdict(deployment)
        })
        
        return jsonify({'status': 'rolled_back', 'deployment_id': deployment_id})
    return jsonify({'error': 'Deployment not found'}), 404

@app.route('/api/regions')
def api_regions():
    """Get regions status API"""
    regions = {k: asdict(v) for k, v in dashboard.region_status.items()}
    return jsonify(regions)

@app.route('/api/chaos')
def api_chaos_experiments():
    """Get chaos experiments API"""
    experiments = {k: asdict(v) for k, v in dashboard.chaos_experiments.items()}
    return jsonify(experiments)

@app.route('/api/chaos/<experiment_id>/stop', methods=['POST'])
def api_stop_chaos_experiment(experiment_id):
    """Stop chaos experiment"""
    if experiment_id in dashboard.chaos_experiments:
        experiment = dashboard.chaos_experiments[experiment_id]
        experiment.status = 'stopped'
        
        # Emit real-time update
        socketio.emit('experiment_updated', {
            'experiment_id': experiment_id,
            'data': asdict(experiment)
        })
        
        return jsonify({'status': 'stopped', 'experiment_id': experiment_id})
    return jsonify({'error': 'Experiment not found'}), 404

@app.route('/api/alerts')
def api_alerts():
    """Get alerts API"""
    return jsonify(dashboard.alerts)

@app.route('/api/charts/timeline')
def api_timeline_chart():
    """Get deployment timeline chart"""
    chart_data = dashboard.create_deployment_timeline_chart()
    return jsonify({'chart': chart_data})

@app.route('/api/charts/metrics')
def api_metrics_chart():
    """Get metrics chart"""
    chart_data = dashboard.create_metrics_chart()
    return jsonify({'chart': chart_data})

@app.route('/api/charts/regions')
def api_region_map_chart():
    """Get region map chart"""
    chart_data = dashboard.create_region_map_chart()
    return jsonify({'chart': chart_data})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to DeployX Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_room')
def handle_join_room(data):
    """Handle room joining for targeted updates"""
    room = data.get('room', 'general')
    join_room(room)
    emit('joined_room', {'room': room})

@socketio.on('leave_room')
def handle_leave_room(data):
    """Handle room leaving"""
    room = data.get('room', 'general')
    leave_room(room)
    emit('left_room', {'room': room})

@socketio.on('request_update')
def handle_request_update():
    """Handle manual update request"""
    dashboard_data = dashboard.get_dashboard_data()
    emit('dashboard_update', dashboard_data)

# Background task for real-time updates
def background_updates():
    """Background task to send periodic updates"""
    while True:
        socketio.sleep(5)  # Update every 5 seconds
        
        # Simulate metric updates
        for deployment in dashboard.active_deployments.values():
            if deployment.status == DeploymentStatus.RUNNING:
                # Simulate progress
                deployment.progress = min(100.0, deployment.progress + 2.5)
                if deployment.progress >= 100.0:
                    deployment.status = DeploymentStatus.SUCCESS
        
        # Send updates to all connected clients
        dashboard_data = dashboard.get_dashboard_data()
        socketio.emit('dashboard_update', dashboard_data)

# Create templates directory and basic template
def create_dashboard_template():
    """Create the dashboard HTML template"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    template_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Commander DeployX Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .status-healthy { color: #10b981; }
        .status-warning { color: #f59e0b; }
        .status-critical { color: #ef4444; }
        .status-unknown { color: #6b7280; }
    </style>
</head>
<body class="bg-gray-100">
    <!-- Header -->
    <header class="gradient-bg text-white shadow-lg">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-4">
                    <i class="fas fa-rocket text-3xl"></i>
                    <div>
                        <h1 class="text-2xl font-bold">Commander DeployX</h1>
                        <p class="text-sm opacity-90">Superhuman Deployment Strategist & Resilience Commander</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="text-right">
                        <div class="text-sm opacity-90">Status</div>
                        <div class="font-semibold" id="connection-status">Connected</div>
                    </div>
                    <div class="w-3 h-3 bg-green-400 rounded-full animate-pulse" id="status-indicator"></div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Dashboard -->
    <main class="container mx-auto px-6 py-8">
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="card rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Active Deployments</p>
                        <p class="text-3xl font-bold text-blue-600" id="active-deployments">0</p>
                    </div>
                    <i class="fas fa-rocket text-blue-500 text-2xl"></i>
                </div>
            </div>
            
            <div class="card rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Success Rate</p>
                        <p class="text-3xl font-bold text-green-600" id="success-rate">0%</p>
                    </div>
                    <i class="fas fa-check-circle text-green-500 text-2xl"></i>
                </div>
            </div>
            
            <div class="card rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Healthy Regions</p>
                        <p class="text-3xl font-bold text-purple-600" id="healthy-regions">0/0</p>
                    </div>
                    <i class="fas fa-globe text-purple-500 text-2xl"></i>
                </div>
            </div>
            
            <div class="card rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Active Alerts</p>
                        <p class="text-3xl font-bold text-red-600" id="active-alerts">0</p>
                    </div>
                    <i class="fas fa-exclamation-triangle text-red-500 text-2xl"></i>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="card rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-semibold mb-4">Deployment Timeline</h3>
                <div id="timeline-chart" style="height: 400px;"></div>
            </div>
            
            <div class="card rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-semibold mb-4">System Metrics</h3>
                <div id="metrics-chart" style="height: 400px;"></div>
            </div>
        </div>

        <!-- Deployments and Regions -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Active Deployments -->
            <div class="card rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold">Active Deployments</h3>
                    <button class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors">
                        <i class="fas fa-plus mr-2"></i>New Deployment
                    </button>
                </div>
                <div id="deployments-list" class="space-y-4">
                    <!-- Deployments will be populated here -->
                </div>
            </div>
            
            <!-- Region Status -->
            <div class="card rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-semibold mb-4">Region Status</h3>
                <div id="region-map" style="height: 300px; margin-bottom: 20px;"></div>
                <div id="regions-list" class="space-y-3">
                    <!-- Regions will be populated here -->
                </div>
            </div>
        </div>

        <!-- Chaos Engineering and Alerts -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Chaos Experiments -->
            <div class="card rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold">Chaos Engineering</h3>
                    <button class="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition-colors">
                        <i class="fas fa-flask mr-2"></i>New Experiment
                    </button>
                </div>
                <div id="chaos-experiments" class="space-y-4">
                    <!-- Experiments will be populated here -->
                </div>
            </div>
            
            <!-- Alerts -->
            <div class="card rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-semibold mb-4">Active Alerts</h3>
                <div id="alerts-list" class="space-y-3">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
        </div>
    </main>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Connection status
        socket.on('connect', () => {
            document.getElementById('connection-status').textContent = 'Connected';
            document.getElementById('status-indicator').className = 'w-3 h-3 bg-green-400 rounded-full animate-pulse';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.getElementById('status-indicator').className = 'w-3 h-3 bg-red-400 rounded-full';
        });
        
        // Dashboard updates
        socket.on('dashboard_update', (data) => {
            updateDashboard(data);
        });
        
        // Update dashboard function
        function updateDashboard(data) {
            // Update summary cards
            document.getElementById('active-deployments').textContent = data.summary.running_deployments;
            document.getElementById('success-rate').textContent = data.summary.avg_success_rate + '%';
            document.getElementById('healthy-regions').textContent = data.summary.healthy_regions + '/' + data.summary.total_regions;
            document.getElementById('active-alerts').textContent = data.summary.active_alerts;
            
            // Update deployments
            updateDeployments(data.deployments);
            
            // Update regions
            updateRegions(data.regions);
            
            // Update chaos experiments
            updateChaosExperiments(data.chaos_experiments);
            
            // Update alerts
            updateAlerts(data.alerts);
        }
        
        function updateDeployments(deployments) {
            const container = document.getElementById('deployments-list');
            container.innerHTML = '';
            
            Object.values(deployments).forEach(deployment => {
                const statusColor = getStatusColor(deployment.status);
                const progressWidth = deployment.progress;
                
                const deploymentCard = document.createElement('div');
                deploymentCard.className = 'border rounded-lg p-4 bg-gray-50';
                deploymentCard.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <h4 class="font-semibold">${deployment.application}</h4>
                        <span class="px-2 py-1 rounded text-sm ${statusColor}">${deployment.status}</span>
                    </div>
                    <div class="text-sm text-gray-600 mb-2">
                        Version: ${deployment.version} | Environment: ${deployment.environment}
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2 mb-2">
                        <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: ${progressWidth}%"></div>
                    </div>
                    <div class="flex justify-between text-sm text-gray-600">
                        <span>Progress: ${progressWidth.toFixed(1)}%</span>
                        <span>Success Rate: ${deployment.success_rate}%</span>
                    </div>
                    ${deployment.canary_confidence ? `
                        <div class="mt-2 flex justify-between">
                            <button onclick="promoteCanary('${deployment.deployment_id}')" class="bg-green-500 text-white px-3 py-1 rounded text-sm hover:bg-green-600">
                                Promote Canary
                            </button>
                            <button onclick="rollbackDeployment('${deployment.deployment_id}')" class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600">
                                Rollback
                            </button>
                        </div>
                    ` : ''}
                `;
                container.appendChild(deploymentCard);
            });
        }
        
        function updateRegions(regions) {
            const container = document.getElementById('regions-list');
            container.innerHTML = '';
            
            Object.values(regions).forEach(region => {
                const healthColor = getHealthColor(region.health);
                
                const regionCard = document.createElement('div');
                regionCard.className = 'flex items-center justify-between p-3 bg-gray-50 rounded';
                regionCard.innerHTML = `
                    <div>
                        <div class="font-semibold">${region.region}</div>
                        <div class="text-sm text-gray-600">${region.deployments} deployments</div>
                    </div>
                    <div class="text-right">
                        <div class="${healthColor} font-semibold">${region.health}</div>
                        <div class="text-sm text-gray-600">${region.availability}% uptime</div>
                    </div>
                `;
                container.appendChild(regionCard);
            });
        }
        
        function updateChaosExperiments(experiments) {
            const container = document.getElementById('chaos-experiments');
            container.innerHTML = '';
            
            Object.values(experiments).forEach(experiment => {
                const statusColor = getStatusColor(experiment.status);
                
                const experimentCard = document.createElement('div');
                experimentCard.className = 'border rounded-lg p-4 bg-gray-50';
                experimentCard.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <h4 class="font-semibold">${experiment.name}</h4>
                        <span class="px-2 py-1 rounded text-sm ${statusColor}">${experiment.status}</span>
                    </div>
                    <div class="text-sm text-gray-600 mb-2">
                        Type: ${experiment.type} | Target: ${experiment.target}
                    </div>
                    ${experiment.resilience_score ? `
                        <div class="text-sm text-gray-600 mb-2">
                            Resilience Score: ${experiment.resilience_score}%
                        </div>
                    ` : ''}
                    ${experiment.status === 'running' ? `
                        <button onclick="stopExperiment('${experiment.experiment_id}')" class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600">
                            Stop Experiment
                        </button>
                    ` : ''}
                `;
                container.appendChild(experimentCard);
            });
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-list');
            container.innerHTML = '';
            
            alerts.forEach(alert => {
                const severityColor = getSeverityColor(alert.severity);
                
                const alertCard = document.createElement('div');
                alertCard.className = 'border-l-4 p-3 bg-gray-50 rounded-r';
                alertCard.style.borderLeftColor = severityColor;
                alertCard.innerHTML = `
                    <div class="flex items-center justify-between mb-1">
                        <h4 class="font-semibold">${alert.title}</h4>
                        <span class="text-xs text-gray-500">${new Date(alert.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div class="text-sm text-gray-600">${alert.message}</div>
                `;
                container.appendChild(alertCard);
            });
        }
        
        // Utility functions
        function getStatusColor(status) {
            const colors = {
                'pending': 'bg-yellow-100 text-yellow-800',
                'running': 'bg-blue-100 text-blue-800',
                'success': 'bg-green-100 text-green-800',
                'failed': 'bg-red-100 text-red-800',
                'rolled_back': 'bg-gray-100 text-gray-800'
            };
            return colors[status] || 'bg-gray-100 text-gray-800';
        }
        
        function getHealthColor(health) {
            const colors = {
                'healthy': 'status-healthy',
                'warning': 'status-warning',
                'critical': 'status-critical',
                'unknown': 'status-unknown'
            };
            return colors[health] || 'status-unknown';
        }
        
        function getSeverityColor(severity) {
            const colors = {
                'info': '#3b82f6',
                'warning': '#f59e0b',
                'critical': '#ef4444'
            };
            return colors[severity] || '#6b7280';
        }
        
        // Action functions
        function promoteCanary(deploymentId) {
            fetch(`/api/deployments/${deploymentId}/promote`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Canary promoted:', data);
                })
                .catch(error => {
                    console.error('Error promoting canary:', error);
                });
        }
        
        function rollbackDeployment(deploymentId) {
            if (confirm('Are you sure you want to rollback this deployment?')) {
                fetch(`/api/deployments/${deploymentId}/rollback`, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Deployment rolled back:', data);
                    })
                    .catch(error => {
                        console.error('Error rolling back deployment:', error);
                    });
            }
        }
        
        function stopExperiment(experimentId) {
            if (confirm('Are you sure you want to stop this chaos experiment?')) {
                fetch(`/api/chaos/${experimentId}/stop`, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Experiment stopped:', data);
                    })
                    .catch(error => {
                        console.error('Error stopping experiment:', error);
                    });
            }
        }
        
        // Load charts
        function loadCharts() {
            // Timeline chart
            fetch('/api/charts/timeline')
                .then(response => response.json())
                .then(data => {
                    if (data.chart && Object.keys(JSON.parse(data.chart)).length > 0) {
                        Plotly.newPlot('timeline-chart', JSON.parse(data.chart));
                    }
                })
                .catch(error => console.error('Error loading timeline chart:', error));
            
            // Metrics chart
            fetch('/api/charts/metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.chart && Object.keys(JSON.parse(data.chart)).length > 0) {
                        Plotly.newPlot('metrics-chart', JSON.parse(data.chart));
                    }
                })
                .catch(error => console.error('Error loading metrics chart:', error));
            
            // Region map
            fetch('/api/charts/regions')
                .then(response => response.json())
                .then(data => {
                    if (data.chart && Object.keys(JSON.parse(data.chart)).length > 0) {
                        Plotly.newPlot('region-map', JSON.parse(data.chart));
                    }
                })
                .catch(error => console.error('Error loading region map:', error));
        }
        
        // Initial load
        fetch('/api/dashboard')
            .then(response => response.json())
            .then(data => {
                updateDashboard(data);
                loadCharts();
            })
            .catch(error => console.error('Error loading dashboard:', error));
        
        // Request periodic updates
        setInterval(() => {
            socket.emit('request_update');
        }, 10000); // Every 10 seconds
    </script>
</body>
</html>
    '''
    
    with open(templates_dir / 'dashboard.html', 'w') as f:
        f.write(template_content)

if __name__ == '__main__':
    # Create template if it doesn't exist
    create_dashboard_template()
    
    # Start background task
    socketio.start_background_task(background_updates)
    
    # Run the application
    logger.info("Starting Commander DeployX Web Dashboard")
    logger.info("Dashboard will be available at: http://localhost:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)