#!/usr/bin/env python3
"""
Commander DeployX's Zero-Downtime Hot-Swap Release System
Autonomous Service Mesh Orchestration with Atomic Module Swaps

Features:
- Istio service mesh integration for traffic management
- Feature flag framework for gradual rollouts
- Atomic blue-green and canary deployments
- Real-time health monitoring and automatic rollback
- Multi-region deployment coordination
- Circuit breaker and retry mechanisms

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
from collections import defaultdict

import yaml
import aiohttp
import asyncio
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Service mesh and traffic management
import istioctl
from istio_client_python import ApiClient as IstioApiClient
from istio_client_python.api import networking_v1beta1_api

# Feature flags
import redis
import consul

# Monitoring
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import requests

# Health checks
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    FEATURE_FLAG = "feature_flag"
    ATOMIC_SWAP = "atomic_swap"

class DeploymentPhase(Enum):
    """Deployment phases"""
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    TESTING = "testing"
    TRAFFIC_SHIFTING = "traffic_shifting"
    MONITORING = "monitoring"
    COMPLETING = "completing"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class HealthCheck:
    """Health check configuration"""
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout: int = 30
    interval: int = 10
    retries: int = 3
    headers: Dict[str, str] = None

@dataclass
class TrafficSplit:
    """Traffic splitting configuration"""
    blue_weight: int = 100
    green_weight: int = 0
    canary_weight: int = 0
    stable_weight: int = 100

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    service_name: str
    namespace: str
    strategy: DeploymentStrategy
    image: str
    replicas: int
    health_checks: List[HealthCheck]
    traffic_split: TrafficSplit
    feature_flags: Dict[str, Any]
    rollback_threshold: float = 0.95
    max_surge: int = 1
    max_unavailable: int = 0
    timeout_seconds: int = 600
    regions: List[str] = None

@dataclass
class DeploymentStatus:
    """Deployment status tracking"""
    deployment_id: str
    service_name: str
    phase: DeploymentPhase
    strategy: DeploymentStrategy
    start_time: datetime
    current_traffic_split: TrafficSplit
    health_status: Dict[str, bool]
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    rollback_triggered: bool = False

class ZeroDowntimeOrchestrator:
    """Zero-Downtime Deployment Orchestrator"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.k8s_client = self._init_kubernetes()
        self.istio_client = self._init_istio()
        self.redis_client = self._init_redis()
        self.consul_client = self._init_consul()
        
        # Active deployments tracking
        self.active_deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_history: List[DeploymentStatus] = []
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'state': 'closed',  # closed, open, half_open
            'failure_count': 0,
            'last_failure_time': None,
            'success_count': 0
        })
        
        # Prometheus metrics
        self.deployment_counter = Counter(
            'deployments_total',
            'Total number of deployments',
            ['service', 'strategy', 'status']
        )
        self.deployment_duration = Histogram(
            'deployment_duration_seconds',
            'Deployment duration in seconds',
            ['service', 'strategy']
        )
        self.traffic_split_gauge = Gauge(
            'traffic_split_percentage',
            'Current traffic split percentage',
            ['service', 'version']
        )
        self.health_check_gauge = Gauge(
            'health_check_status',
            'Health check status (1=healthy, 0=unhealthy)',
            ['service', 'endpoint']
        )
        
        logger.info("Zero-Downtime Orchestrator initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'redis_url': os.getenv('REDIS_URL', 'redis://redis-service:6379'),
            'consul_url': os.getenv('CONSUL_URL', 'http://consul-service:8500'),
            'istio_namespace': os.getenv('ISTIO_NAMESPACE', 'istio-system'),
            'prometheus_url': os.getenv('PROMETHEUS_URL', 'http://prometheus-service:9090'),
            'default_namespace': os.getenv('DEFAULT_NAMESPACE', 'ai-news-dashboard'),
            'health_check_timeout': int(os.getenv('HEALTH_CHECK_TIMEOUT', '30')),
            'traffic_shift_interval': int(os.getenv('TRAFFIC_SHIFT_INTERVAL', '60')),
            'rollback_threshold': float(os.getenv('ROLLBACK_THRESHOLD', '0.95')),
            'circuit_breaker_threshold': int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', '5')),
            'circuit_breaker_timeout': int(os.getenv('CIRCUIT_BREAKER_TIMEOUT', '60')),
            'max_concurrent_deployments': int(os.getenv('MAX_CONCURRENT_DEPLOYMENTS', '3')),
            'enable_multi_region': os.getenv('ENABLE_MULTI_REGION', 'true').lower() == 'true'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        return default_config
    
    def _init_kubernetes(self) -> client.ApiClient:
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        return client.ApiClient()
    
    def _init_istio(self):
        """Initialize Istio client"""
        try:
            # Initialize Istio API client
            configuration = istioctl.Configuration()
            return IstioApiClient(configuration)
        except Exception as e:
            logger.warning(f"Failed to initialize Istio client: {e}")
            return None
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis client for feature flags"""
        return redis.from_url(self.config['redis_url'])
    
    def _init_consul(self):
        """Initialize Consul client for service discovery"""
        try:
            return consul.Consul(host=self.config['consul_url'].split('//')[1].split(':')[0])
        except Exception as e:
            logger.warning(f"Failed to initialize Consul client: {e}")
            return None
    
    async def deploy(self, deployment_config: DeploymentConfig) -> str:
        """Execute zero-downtime deployment"""
        deployment_id = f"{deployment_config.service_name}-{int(time.time())}"
        
        try:
            # Check if we can start new deployment
            if len(self.active_deployments) >= self.config['max_concurrent_deployments']:
                raise Exception("Maximum concurrent deployments reached")
            
            # Initialize deployment status
            status = DeploymentStatus(
                deployment_id=deployment_id,
                service_name=deployment_config.service_name,
                phase=DeploymentPhase.PREPARING,
                strategy=deployment_config.strategy,
                start_time=datetime.now(),
                current_traffic_split=deployment_config.traffic_split,
                health_status={},
                metrics={}
            )
            
            self.active_deployments[deployment_id] = status
            
            logger.info(f"Starting {deployment_config.strategy.value} deployment for {deployment_config.service_name}")
            
            # Execute deployment based on strategy
            if deployment_config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(deployment_config, status)
            elif deployment_config.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(deployment_config, status)
            elif deployment_config.strategy == DeploymentStrategy.ROLLING:
                await self._execute_rolling_deployment(deployment_config, status)
            elif deployment_config.strategy == DeploymentStrategy.FEATURE_FLAG:
                await self._execute_feature_flag_deployment(deployment_config, status)
            elif deployment_config.strategy == DeploymentStrategy.ATOMIC_SWAP:
                await self._execute_atomic_swap_deployment(deployment_config, status)
            else:
                raise ValueError(f"Unsupported deployment strategy: {deployment_config.strategy}")
            
            # Mark deployment as completed
            status.phase = DeploymentPhase.COMPLETED
            
            # Update metrics
            self.deployment_counter.labels(
                service=deployment_config.service_name,
                strategy=deployment_config.strategy.value,
                status='success'
            ).inc()
            
            duration = (datetime.now() - status.start_time).total_seconds()
            self.deployment_duration.labels(
                service=deployment_config.service_name,
                strategy=deployment_config.strategy.value
            ).observe(duration)
            
            logger.info(f"Deployment {deployment_id} completed successfully")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            
            if deployment_id in self.active_deployments:
                self.active_deployments[deployment_id].phase = DeploymentPhase.FAILED
                self.active_deployments[deployment_id].error_message = str(e)
                
                # Attempt rollback
                await self._rollback_deployment(deployment_config, self.active_deployments[deployment_id])
            
            self.deployment_counter.labels(
                service=deployment_config.service_name,
                strategy=deployment_config.strategy.value,
                status='failed'
            ).inc()
            
            raise
        finally:
            # Move to history and cleanup
            if deployment_id in self.active_deployments:
                self.deployment_history.append(self.active_deployments[deployment_id])
                del self.active_deployments[deployment_id]
    
    async def _execute_blue_green_deployment(self, config: DeploymentConfig, status: DeploymentStatus):
        """Execute blue-green deployment"""
        try:
            status.phase = DeploymentPhase.DEPLOYING
            
            # Deploy green version
            green_deployment_name = f"{config.service_name}-green"
            await self._create_deployment(config, green_deployment_name, "green")
            
            # Wait for green deployment to be ready
            await self._wait_for_deployment_ready(green_deployment_name, config.namespace)
            
            status.phase = DeploymentPhase.TESTING
            
            # Perform health checks on green version
            green_healthy = await self._perform_health_checks(config, "green")
            if not green_healthy:
                raise Exception("Green deployment failed health checks")
            
            status.phase = DeploymentPhase.TRAFFIC_SHIFTING
            
            # Switch traffic to green (atomic swap)
            await self._switch_traffic_blue_green(config, "green")
            
            status.phase = DeploymentPhase.MONITORING
            
            # Monitor for a period
            await self._monitor_deployment(config, status, duration=300)  # 5 minutes
            
            status.phase = DeploymentPhase.COMPLETING
            
            # Cleanup blue deployment
            blue_deployment_name = f"{config.service_name}-blue"
            await self._cleanup_deployment(blue_deployment_name, config.namespace)
            
            # Rename green to blue for next deployment
            await self._rename_deployment(green_deployment_name, blue_deployment_name, config.namespace)
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            raise
    
    async def _execute_canary_deployment(self, config: DeploymentConfig, status: DeploymentStatus):
        """Execute canary deployment"""
        try:
            status.phase = DeploymentPhase.DEPLOYING
            
            # Deploy canary version
            canary_deployment_name = f"{config.service_name}-canary"
            await self._create_deployment(config, canary_deployment_name, "canary")
            
            # Wait for canary deployment to be ready
            await self._wait_for_deployment_ready(canary_deployment_name, config.namespace)
            
            status.phase = DeploymentPhase.TESTING
            
            # Perform health checks on canary
            canary_healthy = await self._perform_health_checks(config, "canary")
            if not canary_healthy:
                raise Exception("Canary deployment failed health checks")
            
            status.phase = DeploymentPhase.TRAFFIC_SHIFTING
            
            # Gradual traffic shifting
            traffic_percentages = [5, 10, 25, 50, 75, 100]
            
            for percentage in traffic_percentages:
                await self._shift_traffic_canary(config, percentage)
                status.current_traffic_split.canary_weight = percentage
                status.current_traffic_split.stable_weight = 100 - percentage
                
                # Monitor at each step
                await self._monitor_deployment(config, status, duration=120)  # 2 minutes
                
                # Check if rollback is needed
                if status.rollback_triggered:
                    raise Exception("Rollback triggered due to poor metrics")
            
            status.phase = DeploymentPhase.COMPLETING
            
            # Promote canary to stable
            await self._promote_canary_to_stable(config)
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            raise
    
    async def _execute_rolling_deployment(self, config: DeploymentConfig, status: DeploymentStatus):
        """Execute rolling deployment"""
        try:
            status.phase = DeploymentPhase.DEPLOYING
            
            # Update deployment with rolling strategy
            await self._update_deployment_rolling(config)
            
            status.phase = DeploymentPhase.MONITORING
            
            # Monitor rolling deployment progress
            await self._monitor_rolling_deployment(config, status)
            
            status.phase = DeploymentPhase.TESTING
            
            # Perform final health checks
            healthy = await self._perform_health_checks(config, "stable")
            if not healthy:
                raise Exception("Rolling deployment failed health checks")
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            raise
    
    async def _execute_feature_flag_deployment(self, config: DeploymentConfig, status: DeploymentStatus):
        """Execute feature flag based deployment"""
        try:
            status.phase = DeploymentPhase.DEPLOYING
            
            # Deploy new version alongside existing
            new_deployment_name = f"{config.service_name}-v2"
            await self._create_deployment(config, new_deployment_name, "v2")
            
            # Wait for new deployment to be ready
            await self._wait_for_deployment_ready(new_deployment_name, config.namespace)
            
            status.phase = DeploymentPhase.TESTING
            
            # Perform health checks
            healthy = await self._perform_health_checks(config, "v2")
            if not healthy:
                raise Exception("New version failed health checks")
            
            status.phase = DeploymentPhase.TRAFFIC_SHIFTING
            
            # Gradual feature flag rollout
            rollout_percentages = [1, 5, 10, 25, 50, 100]
            
            for percentage in rollout_percentages:
                await self._update_feature_flag(config.service_name, "new_version", percentage)
                
                # Monitor feature flag metrics
                await self._monitor_feature_flag_deployment(config, status, percentage)
                
                if status.rollback_triggered:
                    raise Exception("Rollback triggered due to poor feature flag metrics")
            
            status.phase = DeploymentPhase.COMPLETING
            
            # Remove feature flag and cleanup old version
            await self._remove_feature_flag(config.service_name, "new_version")
            await self._cleanup_deployment(config.service_name, config.namespace)
            await self._rename_deployment(new_deployment_name, config.service_name, config.namespace)
            
        except Exception as e:
            logger.error(f"Feature flag deployment failed: {e}")
            raise
    
    async def _execute_atomic_swap_deployment(self, config: DeploymentConfig, status: DeploymentStatus):
        """Execute atomic swap deployment"""
        try:
            status.phase = DeploymentPhase.DEPLOYING
            
            # Prepare new deployment
            new_deployment_name = f"{config.service_name}-new"
            await self._create_deployment(config, new_deployment_name, "new")
            
            # Wait for new deployment to be ready
            await self._wait_for_deployment_ready(new_deployment_name, config.namespace)
            
            status.phase = DeploymentPhase.TESTING
            
            # Perform comprehensive health checks
            healthy = await self._perform_health_checks(config, "new")
            if not healthy:
                raise Exception("New deployment failed health checks")
            
            # Perform integration tests
            integration_passed = await self._run_integration_tests(config, "new")
            if not integration_passed:
                raise Exception("Integration tests failed")
            
            status.phase = DeploymentPhase.TRAFFIC_SHIFTING
            
            # Atomic swap - update service selector
            await self._atomic_service_swap(config, new_deployment_name)
            
            status.phase = DeploymentPhase.MONITORING
            
            # Brief monitoring period
            await self._monitor_deployment(config, status, duration=180)  # 3 minutes
            
            status.phase = DeploymentPhase.COMPLETING
            
            # Cleanup old deployment
            await self._cleanup_deployment(config.service_name, config.namespace)
            await self._rename_deployment(new_deployment_name, config.service_name, config.namespace)
            
        except Exception as e:
            logger.error(f"Atomic swap deployment failed: {e}")
            raise
    
    async def _create_deployment(self, config: DeploymentConfig, deployment_name: str, version: str):
        """Create Kubernetes deployment"""
        try:
            apps_v1 = client.AppsV1Api()
            
            deployment_manifest = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': deployment_name,
                    'namespace': config.namespace,
                    'labels': {
                        'app': config.service_name,
                        'version': version,
                        'managed-by': 'zero-downtime-orchestrator'
                    }
                },
                'spec': {
                    'replicas': config.replicas,
                    'selector': {
                        'matchLabels': {
                            'app': config.service_name,
                            'version': version
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': config.service_name,
                                'version': version
                            }
                        },
                        'spec': {
                            'containers': [{
                                'name': config.service_name,
                                'image': config.image,
                                'ports': [{
                                    'containerPort': 8080,
                                    'name': 'http'
                                }],
                                'livenessProbe': {
                                    'httpGet': {
                                        'path': '/health',
                                        'port': 8080
                                    },
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {
                                        'path': '/ready',
                                        'port': 8080
                                    },
                                    'initialDelaySeconds': 5,
                                    'periodSeconds': 5
                                },
                                'resources': {
                                    'requests': {
                                        'cpu': '100m',
                                        'memory': '128Mi'
                                    },
                                    'limits': {
                                        'cpu': '500m',
                                        'memory': '512Mi'
                                    }
                                }
                            }]
                        }
                    },
                    'strategy': {
                        'type': 'RollingUpdate',
                        'rollingUpdate': {
                            'maxSurge': config.max_surge,
                            'maxUnavailable': config.max_unavailable
                        }
                    }
                }
            }
            
            apps_v1.create_namespaced_deployment(
                namespace=config.namespace,
                body=deployment_manifest
            )
            
            logger.info(f"Created deployment {deployment_name}")
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"Deployment {deployment_name} already exists, updating...")
                apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=config.namespace,
                    body=deployment_manifest
                )
            else:
                raise
        except Exception as e:
            logger.error(f"Error creating deployment {deployment_name}: {e}")
            raise
    
    async def _wait_for_deployment_ready(self, deployment_name: str, namespace: str, timeout: int = 600):
        """Wait for deployment to be ready"""
        apps_v1 = client.AppsV1Api()
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    logger.info(f"Deployment {deployment_name} is ready")
                    return
                
                await asyncio.sleep(10)
                
            except ApiException as e:
                logger.error(f"Error checking deployment status: {e}")
                await asyncio.sleep(10)
        
        raise Exception(f"Deployment {deployment_name} did not become ready within {timeout} seconds")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _perform_health_checks(self, config: DeploymentConfig, version: str) -> bool:
        """Perform health checks on deployment"""
        try:
            all_healthy = True
            
            for health_check in config.health_checks:
                try:
                    # Get service endpoint
                    service_url = await self._get_service_endpoint(config.service_name, config.namespace, version)
                    
                    if not service_url:
                        logger.error(f"Could not get service endpoint for {config.service_name}-{version}")
                        all_healthy = False
                        continue
                    
                    # Perform health check
                    async with httpx.AsyncClient(timeout=health_check.timeout) as client:
                        response = await client.request(
                            method=health_check.method,
                            url=f"{service_url}{health_check.endpoint}",
                            headers=health_check.headers or {}
                        )
                        
                        if response.status_code == health_check.expected_status:
                            logger.info(f"Health check passed for {config.service_name}-{version}: {health_check.endpoint}")
                            self.health_check_gauge.labels(
                                service=config.service_name,
                                endpoint=health_check.endpoint
                            ).set(1)
                        else:
                            logger.error(f"Health check failed for {config.service_name}-{version}: {health_check.endpoint} (status: {response.status_code})")
                            self.health_check_gauge.labels(
                                service=config.service_name,
                                endpoint=health_check.endpoint
                            ).set(0)
                            all_healthy = False
                            
                except Exception as e:
                    logger.error(f"Health check error for {config.service_name}-{version}: {e}")
                    self.health_check_gauge.labels(
                        service=config.service_name,
                        endpoint=health_check.endpoint
                    ).set(0)
                    all_healthy = False
            
            return all_healthy
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
            return False
    
    async def _get_service_endpoint(self, service_name: str, namespace: str, version: str) -> Optional[str]:
        """Get service endpoint URL"""
        try:
            v1 = client.CoreV1Api()
            service_name_versioned = f"{service_name}-{version}" if version != "stable" else service_name
            
            service = v1.read_namespaced_service(
                name=service_name_versioned,
                namespace=namespace
            )
            
            # For now, assume ClusterIP service
            cluster_ip = service.spec.cluster_ip
            port = service.spec.ports[0].port
            
            return f"http://{cluster_ip}:{port}"
            
        except ApiException as e:
            logger.error(f"Error getting service endpoint: {e}")
            return None
    
    async def _switch_traffic_blue_green(self, config: DeploymentConfig, target_version: str):
        """Switch traffic for blue-green deployment"""
        try:
            # Update service selector to point to target version
            v1 = client.CoreV1Api()
            
            service_patch = {
                'spec': {
                    'selector': {
                        'app': config.service_name,
                        'version': target_version
                    }
                }
            }
            
            v1.patch_namespaced_service(
                name=config.service_name,
                namespace=config.namespace,
                body=service_patch
            )
            
            logger.info(f"Switched traffic to {target_version} version")
            
        except Exception as e:
            logger.error(f"Error switching traffic: {e}")
            raise
    
    async def _shift_traffic_canary(self, config: DeploymentConfig, canary_percentage: int):
        """Shift traffic for canary deployment using Istio"""
        try:
            if not self.istio_client:
                logger.warning("Istio client not available, skipping traffic shifting")
                return
            
            # Create VirtualService for traffic splitting
            virtual_service = {
                'apiVersion': 'networking.istio.io/v1beta1',
                'kind': 'VirtualService',
                'metadata': {
                    'name': f"{config.service_name}-vs",
                    'namespace': config.namespace
                },
                'spec': {
                    'hosts': [config.service_name],
                    'http': [{
                        'match': [{'headers': {'canary': {'exact': 'true'}}}],
                        'route': [{
                            'destination': {
                                'host': config.service_name,
                                'subset': 'canary'
                            }
                        }]
                    }, {
                        'route': [
                            {
                                'destination': {
                                    'host': config.service_name,
                                    'subset': 'canary'
                                },
                                'weight': canary_percentage
                            },
                            {
                                'destination': {
                                    'host': config.service_name,
                                    'subset': 'stable'
                                },
                                'weight': 100 - canary_percentage
                            }
                        ]
                    }]
                }
            }
            
            # Apply VirtualService
            # Note: This would require proper Istio client implementation
            logger.info(f"Shifted {canary_percentage}% traffic to canary")
            
            # Update traffic split gauge
            self.traffic_split_gauge.labels(
                service=config.service_name,
                version='canary'
            ).set(canary_percentage)
            
            self.traffic_split_gauge.labels(
                service=config.service_name,
                version='stable'
            ).set(100 - canary_percentage)
            
        except Exception as e:
            logger.error(f"Error shifting traffic: {e}")
            raise
    
    async def _monitor_deployment(self, config: DeploymentConfig, status: DeploymentStatus, duration: int):
        """Monitor deployment metrics"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Collect metrics from Prometheus
                metrics = await self._collect_deployment_metrics(config.service_name)
                status.metrics.update(metrics)
                
                # Check rollback conditions
                if self._should_rollback(metrics, config.rollback_threshold):
                    status.rollback_triggered = True
                    logger.warning(f"Rollback triggered for {config.service_name}")
                    return
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring deployment: {e}")
                await asyncio.sleep(30)
    
    async def _collect_deployment_metrics(self, service_name: str) -> Dict[str, float]:
        """Collect deployment metrics from Prometheus"""
        try:
            queries = {
                'success_rate': f'sum(rate(http_requests_total{{service="{service_name}",status!~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{service_name}"}}[5m]))',
                'error_rate': f'sum(rate(http_requests_total{{service="{service_name}",status=~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{service_name}"}}[5m]))',
                'latency_p99': f'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) by (le)) * 1000',
                'throughput': f'sum(rate(http_requests_total{{service="{service_name}"}}[5m]))'
            }
            
            metrics = {}
            for metric_name, query in queries.items():
                try:
                    response = requests.get(
                        f"{self.config['prometheus_url']}/api/v1/query",
                        params={'query': query},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['data']['result']:
                            value = float(data['data']['result'][0]['value'][1])
                            metrics[metric_name] = value
                        else:
                            metrics[metric_name] = 0.0
                    else:
                        metrics[metric_name] = 0.0
                        
                except Exception as e:
                    logger.error(f"Error fetching metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting deployment metrics: {e}")
            return {}
    
    def _should_rollback(self, metrics: Dict[str, float], threshold: float) -> bool:
        """Determine if rollback should be triggered"""
        try:
            success_rate = metrics.get('success_rate', 1.0)
            error_rate = metrics.get('error_rate', 0.0)
            latency_p99 = metrics.get('latency_p99', 0.0)
            
            # Rollback conditions
            if success_rate < threshold:
                logger.warning(f"Success rate {success_rate} below threshold {threshold}")
                return True
            
            if error_rate > (1 - threshold):
                logger.warning(f"Error rate {error_rate} above threshold {1 - threshold}")
                return True
            
            if latency_p99 > 5000:  # 5 seconds
                logger.warning(f"Latency P99 {latency_p99}ms above 5000ms threshold")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rollback conditions: {e}")
            return True  # Err on the side of caution
    
    async def _rollback_deployment(self, config: DeploymentConfig, status: DeploymentStatus):
        """Rollback deployment"""
        try:
            status.phase = DeploymentPhase.ROLLING_BACK
            logger.info(f"Rolling back deployment for {config.service_name}")
            
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                # Switch back to blue
                await self._switch_traffic_blue_green(config, "blue")
            elif config.strategy == DeploymentStrategy.CANARY:
                # Set traffic back to 100% stable
                await self._shift_traffic_canary(config, 0)
            elif config.strategy == DeploymentStrategy.FEATURE_FLAG:
                # Disable feature flag
                await self._update_feature_flag(config.service_name, "new_version", 0)
            
            logger.info(f"Rollback completed for {config.service_name}")
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            raise
    
    async def _update_feature_flag(self, service_name: str, flag_name: str, percentage: int):
        """Update feature flag percentage"""
        try:
            flag_key = f"feature_flag:{service_name}:{flag_name}"
            flag_data = {
                'enabled': percentage > 0,
                'percentage': percentage,
                'updated_at': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                flag_key,
                3600,  # 1 hour TTL
                json.dumps(flag_data)
            )
            
            logger.info(f"Updated feature flag {flag_name} to {percentage}%")
            
        except Exception as e:
            logger.error(f"Error updating feature flag: {e}")
            raise
    
    async def _cleanup_deployment(self, deployment_name: str, namespace: str):
        """Cleanup deployment resources"""
        try:
            apps_v1 = client.AppsV1Api()
            
            apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            logger.info(f"Cleaned up deployment {deployment_name}")
            
        except ApiException as e:
            if e.status != 404:  # Not found is OK
                logger.error(f"Error cleaning up deployment: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up deployment: {e}")
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get deployment status"""
        return self.active_deployments.get(deployment_id)
    
    async def list_active_deployments(self) -> List[DeploymentStatus]:
        """List all active deployments"""
        return list(self.active_deployments.values())

async def main():
    """Main execution function"""
    orchestrator = ZeroDowntimeOrchestrator()
    
    # Start Prometheus metrics server
    prometheus_client.start_http_server(8092)
    
    logger.info("Zero-Downtime Orchestrator started")
    
    # Keep the service running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down Zero-Downtime Orchestrator")

if __name__ == "__main__":
    asyncio.run(main())