#!/usr/bin/env python3
"""
Zero-Downtime Deployment Orchestrator

This module implements zero-downtime hot-swap releases using service meshes,
feature flags, and atomic module swaps. It ensures seamless deployments
without user-visible disruption.

Features:
- Atomic module swaps
- Service mesh integration (Istio)
- Feature flag coordination
- Health check orchestration
- Traffic shifting strategies
- Rollback mechanisms
- Database migration coordination

Author: Commander Solaris "DeployX" Vivante
"""

import asyncio
import logging
import time
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import subprocess
from pathlib import Path
import hashlib
import tempfile
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    ROLLING_UPDATE = "rolling_update"
    CANARY = "canary"
    ATOMIC_SWAP = "atomic_swap"
    FEATURE_FLAG = "feature_flag"

class DeploymentPhase(Enum):
    """Deployment phases"""
    INITIALIZING = "initializing"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    HEALTH_CHECKING = "health_checking"
    TRAFFIC_SHIFTING = "traffic_shifting"
    VALIDATING = "validating"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"

class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class TrafficShiftStrategy(Enum):
    """Traffic shifting strategies"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    WEIGHTED = "weighted"
    HEADER_BASED = "header_based"
    GEOGRAPHIC = "geographic"

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    endpoint: str
    method: str
    expected_status: int
    timeout_seconds: int
    interval_seconds: int
    retries: int
    headers: Dict[str, str]
    body: Optional[str] = None

@dataclass
class TrafficRule:
    """Traffic routing rule"""
    name: str
    match_criteria: Dict[str, Any]
    destination: str
    weight: float
    headers: Dict[str, str]
    priority: int

@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    enabled: bool
    rollout_percentage: float
    target_groups: List[str]
    conditions: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class DatabaseMigration:
    """Database migration configuration"""
    name: str
    version: str
    script_path: str
    rollback_script_path: str
    pre_deployment: bool
    post_deployment: bool
    timeout_minutes: int

@dataclass
class DeploymentTarget:
    """Deployment target configuration"""
    name: str
    namespace: str
    cluster: str
    region: str
    environment: str
    replicas: int
    resources: Dict[str, Any]
    config_maps: List[str]
    secrets: List[str]
    volumes: List[Dict[str, Any]]

@dataclass
class DeploymentResult:
    """Deployment operation result"""
    success: bool
    phase: DeploymentPhase
    message: str
    timestamp: datetime
    duration_seconds: float
    details: Dict[str, Any]
    rollback_info: Optional[Dict[str, Any]] = None

@dataclass
class ZeroDowntimeDeployment:
    """Zero-downtime deployment configuration"""
    id: str
    name: str
    strategy: DeploymentStrategy
    source_version: str
    target_version: str
    targets: List[DeploymentTarget]
    health_checks: List[HealthCheck]
    traffic_rules: List[TrafficRule]
    feature_flags: List[FeatureFlag]
    database_migrations: List[DatabaseMigration]
    rollback_config: Dict[str, Any]
    timeout_minutes: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_phase: DeploymentPhase = DeploymentPhase.INITIALIZING
    results: List[DeploymentResult] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []

class ZeroDowntimeDeployer:
    """Zero-Downtime Deployment Orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Zero-Downtime Deployer"""
        self.config = self._load_config(config_path)
        
        # Kubernetes client
        self.k8s_client = None
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        
        # Service mesh clients
        self.istio_client = None
        
        # Feature flag client
        self.feature_flag_client = None
        
        # Active deployments
        self.active_deployments = {}
        self.deployment_history = []
        
        # Metrics and monitoring
        self.prometheus_url = self.config.get("prometheus_url", "http://localhost:9090")
        
        logger.info("Zero-Downtime Deployer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration for zero-downtime deployments"""
        default_config = {
            "kubernetes": {
                "config_path": None,  # Use default kubeconfig
                "context": None
            },
            "istio": {
                "enabled": True,
                "gateway": "default-gateway",
                "virtual_service_template": "templates/virtual-service.yaml"
            },
            "feature_flags": {
                "enabled": True,
                "provider": "launchdarkly",  # or "flagsmith", "unleash"
                "api_key": None
            },
            "health_checks": {
                "default_timeout": 30,
                "default_interval": 5,
                "default_retries": 3,
                "startup_grace_period": 60
            },
            "traffic_shifting": {
                "strategy": "gradual",
                "steps": [10, 25, 50, 75, 100],
                "step_duration_minutes": 5,
                "validation_duration_minutes": 2
            },
            "rollback": {
                "automatic": True,
                "health_check_failures": 3,
                "error_rate_threshold": 0.05,
                "response_time_threshold": 2.0
            },
            "database": {
                "migration_timeout_minutes": 30,
                "backup_before_migration": True,
                "rollback_on_failure": True
            },
            "monitoring": {
                "prometheus_url": "http://localhost:9090",
                "grafana_url": "http://localhost:3000",
                "alert_manager_url": "http://localhost:9093"
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
        """Initialize the deployer"""
        logger.info("Initializing Zero-Downtime Deployer...")
        
        try:
            # Initialize Kubernetes client
            await self._initialize_kubernetes()
            
            # Initialize service mesh
            await self._initialize_service_mesh()
            
            # Initialize feature flags
            await self._initialize_feature_flags()
            
            # Test monitoring connectivity
            await self._test_monitoring_connection()
            
            logger.info("Zero-Downtime Deployer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Zero-Downtime Deployer: {e}")
            raise
    
    async def _initialize_kubernetes(self):
        """Initialize Kubernetes client"""
        try:
            # Load kubeconfig
            if self.config["kubernetes"]["config_path"]:
                config.load_kube_config(config_file=self.config["kubernetes"]["config_path"])
            else:
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
            
            # Create clients
            self.k8s_client = client.ApiClient()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            
            # Test connection
            version = await asyncio.get_event_loop().run_in_executor(
                None, self.k8s_core_v1.get_api_resources
            )
            
            logger.info("Kubernetes client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            # For demo purposes, continue without K8s
            logger.info("Continuing in demo mode without Kubernetes")
    
    async def _initialize_service_mesh(self):
        """Initialize service mesh integration"""
        if not self.config["istio"]["enabled"]:
            logger.info("Istio integration disabled")
            return
        
        try:
            # In real implementation, initialize Istio client
            # For demo, simulate Istio availability
            logger.info("Istio service mesh integration initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize service mesh: {e}")
    
    async def _initialize_feature_flags(self):
        """Initialize feature flag integration"""
        if not self.config["feature_flags"]["enabled"]:
            logger.info("Feature flags disabled")
            return
        
        try:
            # In real implementation, initialize feature flag client
            # For demo, simulate feature flag availability
            logger.info("Feature flag integration initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize feature flags: {e}")
    
    async def _test_monitoring_connection(self):
        """Test connection to monitoring systems"""
        try:
            # Test Prometheus
            response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                  params={"query": "up"}, timeout=10)
            response.raise_for_status()
            logger.info("Monitoring connection successful")
            
        except Exception as e:
            logger.warning(f"Monitoring connection failed: {e}")
    
    async def deploy(self, deployment: ZeroDowntimeDeployment) -> str:
        """Execute zero-downtime deployment"""
        logger.info(f"Starting zero-downtime deployment: {deployment.name}")
        
        # Store deployment
        self.active_deployments[deployment.id] = deployment
        deployment.started_at = datetime.now()
        
        # Start deployment process
        asyncio.create_task(self._execute_deployment(deployment))
        
        logger.info(f"Zero-downtime deployment started: {deployment.id}")
        return deployment.id
    
    async def _execute_deployment(self, deployment: ZeroDowntimeDeployment):
        """Execute the deployment process"""
        try:
            # Phase 1: Preparation
            await self._phase_prepare(deployment)
            
            # Phase 2: Database migrations (pre-deployment)
            await self._phase_database_migrations(deployment, pre_deployment=True)
            
            # Phase 3: Deploy new version
            await self._phase_deploy(deployment)
            
            # Phase 4: Health checks
            await self._phase_health_checks(deployment)
            
            # Phase 5: Traffic shifting
            await self._phase_traffic_shifting(deployment)
            
            # Phase 6: Validation
            await self._phase_validation(deployment)
            
            # Phase 7: Database migrations (post-deployment)
            await self._phase_database_migrations(deployment, pre_deployment=False)
            
            # Phase 8: Completion
            await self._phase_completion(deployment)
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            await self._handle_deployment_failure(deployment, str(e))
    
    async def _phase_prepare(self, deployment: ZeroDowntimeDeployment):
        """Preparation phase"""
        logger.info(f"Phase: Preparing deployment {deployment.name}")
        deployment.current_phase = DeploymentPhase.PREPARING
        
        start_time = time.time()
        
        try:
            # Validate deployment configuration
            await self._validate_deployment_config(deployment)
            
            # Prepare deployment manifests
            await self._prepare_manifests(deployment)
            
            # Setup feature flags
            await self._setup_feature_flags(deployment)
            
            # Create backup points
            await self._create_backup_points(deployment)
            
            duration = time.time() - start_time
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.PREPARING,
                message="Preparation completed successfully",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"manifests_prepared": True, "backups_created": True}
            )
            deployment.results.append(result)
            
            logger.info(f"Preparation phase completed in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.PREPARING,
                message=f"Preparation failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"error": str(e)}
            )
            deployment.results.append(result)
            raise
    
    async def _validate_deployment_config(self, deployment: ZeroDowntimeDeployment):
        """Validate deployment configuration"""
        logger.info("Validating deployment configuration...")
        
        # Validate targets
        if not deployment.targets:
            raise ValueError("No deployment targets specified")
        
        # Validate health checks
        if not deployment.health_checks:
            logger.warning("No health checks specified")
        
        # Validate strategy-specific requirements
        if deployment.strategy == DeploymentStrategy.BLUE_GREEN:
            # Blue-green requires specific traffic rules
            pass
        elif deployment.strategy == DeploymentStrategy.CANARY:
            # Canary requires gradual traffic shifting
            pass
        
        logger.info("Deployment configuration validated")
    
    async def _prepare_manifests(self, deployment: ZeroDowntimeDeployment):
        """Prepare Kubernetes manifests"""
        logger.info("Preparing deployment manifests...")
        
        for target in deployment.targets:
            # Generate deployment manifest
            manifest = self._generate_deployment_manifest(deployment, target)
            
            # Generate service manifest
            service_manifest = self._generate_service_manifest(deployment, target)
            
            # Generate Istio virtual service
            if self.config["istio"]["enabled"]:
                vs_manifest = self._generate_virtual_service_manifest(deployment, target)
        
        logger.info("Deployment manifests prepared")
    
    def _generate_deployment_manifest(self, deployment: ZeroDowntimeDeployment, 
                                    target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{deployment.name}-{target.name}",
                "namespace": target.namespace,
                "labels": {
                    "app": deployment.name,
                    "version": deployment.target_version,
                    "deployment-id": deployment.id
                }
            },
            "spec": {
                "replicas": target.replicas,
                "selector": {
                    "matchLabels": {
                        "app": deployment.name,
                        "version": deployment.target_version
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": deployment.name,
                            "version": deployment.target_version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": deployment.name,
                            "image": f"{deployment.name}:{deployment.target_version}",
                            "resources": target.resources,
                            "ports": [{"containerPort": 8080}]
                        }]
                    }
                }
            }
        }
    
    def _generate_service_manifest(self, deployment: ZeroDowntimeDeployment, 
                                 target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{deployment.name}-{target.name}",
                "namespace": target.namespace
            },
            "spec": {
                "selector": {
                    "app": deployment.name,
                    "version": deployment.target_version
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8080
                }]
            }
        }
    
    def _generate_virtual_service_manifest(self, deployment: ZeroDowntimeDeployment, 
                                         target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Istio VirtualService manifest"""
        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{deployment.name}-vs",
                "namespace": target.namespace
            },
            "spec": {
                "hosts": [deployment.name],
                "http": [{
                    "route": [{
                        "destination": {
                            "host": deployment.name,
                            "subset": "v1"
                        },
                        "weight": 100
                    }]
                }]
            }
        }
    
    async def _setup_feature_flags(self, deployment: ZeroDowntimeDeployment):
        """Setup feature flags for deployment"""
        if not self.config["feature_flags"]["enabled"]:
            return
        
        logger.info("Setting up feature flags...")
        
        for flag in deployment.feature_flags:
            # In real implementation, create/update feature flags
            logger.info(f"Feature flag configured: {flag.name} = {flag.enabled}")
        
        logger.info("Feature flags setup completed")
    
    async def _create_backup_points(self, deployment: ZeroDowntimeDeployment):
        """Create backup points for rollback"""
        logger.info("Creating backup points...")
        
        # In real implementation, create snapshots of current state
        backup_info = {
            "timestamp": datetime.now().isoformat(),
            "source_version": deployment.source_version,
            "targets": [target.name for target in deployment.targets]
        }
        
        deployment.rollback_config["backup_info"] = backup_info
        
        logger.info("Backup points created")
    
    async def _phase_database_migrations(self, deployment: ZeroDowntimeDeployment, 
                                       pre_deployment: bool):
        """Database migration phase"""
        phase_name = "pre-deployment" if pre_deployment else "post-deployment"
        logger.info(f"Phase: Database migrations ({phase_name})")
        
        migrations = [m for m in deployment.database_migrations 
                     if m.pre_deployment == pre_deployment]
        
        if not migrations:
            logger.info(f"No {phase_name} migrations to execute")
            return
        
        start_time = time.time()
        
        try:
            for migration in migrations:
                await self._execute_migration(migration)
            
            duration = time.time() - start_time
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.DEPLOYING,
                message=f"{phase_name.title()} migrations completed",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"migrations_executed": len(migrations)}
            )
            deployment.results.append(result)
            
        except Exception as e:
            duration = time.time() - start_time
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.DEPLOYING,
                message=f"{phase_name.title()} migrations failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"error": str(e)}
            )
            deployment.results.append(result)
            raise
    
    async def _execute_migration(self, migration: DatabaseMigration):
        """Execute a database migration"""
        logger.info(f"Executing migration: {migration.name}")
        
        # In real implementation, execute migration script
        # For demo, simulate migration
        await asyncio.sleep(2)
        
        logger.info(f"Migration completed: {migration.name}")
    
    async def _phase_deploy(self, deployment: ZeroDowntimeDeployment):
        """Deployment phase"""
        logger.info(f"Phase: Deploying {deployment.name}")
        deployment.current_phase = DeploymentPhase.DEPLOYING
        
        start_time = time.time()
        
        try:
            for target in deployment.targets:
                await self._deploy_to_target(deployment, target)
            
            duration = time.time() - start_time
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.DEPLOYING,
                message="Deployment completed successfully",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"targets_deployed": len(deployment.targets)}
            )
            deployment.results.append(result)
            
        except Exception as e:
            duration = time.time() - start_time
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.DEPLOYING,
                message=f"Deployment failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"error": str(e)}
            )
            deployment.results.append(result)
            raise
    
    async def _deploy_to_target(self, deployment: ZeroDowntimeDeployment, 
                              target: DeploymentTarget):
        """Deploy to a specific target"""
        logger.info(f"Deploying to target: {target.name}")
        
        if deployment.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._deploy_blue_green(deployment, target)
        elif deployment.strategy == DeploymentStrategy.ROLLING_UPDATE:
            await self._deploy_rolling_update(deployment, target)
        elif deployment.strategy == DeploymentStrategy.ATOMIC_SWAP:
            await self._deploy_atomic_swap(deployment, target)
        else:
            await self._deploy_standard(deployment, target)
        
        logger.info(f"Deployment to {target.name} completed")
    
    async def _deploy_blue_green(self, deployment: ZeroDowntimeDeployment, 
                               target: DeploymentTarget):
        """Execute blue-green deployment"""
        logger.info(f"Executing blue-green deployment to {target.name}")
        
        # Deploy green environment
        green_name = f"{deployment.name}-green"
        
        # In real implementation, deploy to Kubernetes
        # For demo, simulate deployment
        await asyncio.sleep(3)
        
        logger.info(f"Green environment deployed: {green_name}")
    
    async def _deploy_rolling_update(self, deployment: ZeroDowntimeDeployment, 
                                   target: DeploymentTarget):
        """Execute rolling update deployment"""
        logger.info(f"Executing rolling update to {target.name}")
        
        # Update deployment with rolling strategy
        # In real implementation, update Kubernetes deployment
        # For demo, simulate rolling update
        await asyncio.sleep(4)
        
        logger.info(f"Rolling update completed: {target.name}")
    
    async def _deploy_atomic_swap(self, deployment: ZeroDowntimeDeployment, 
                                target: DeploymentTarget):
        """Execute atomic swap deployment"""
        logger.info(f"Executing atomic swap to {target.name}")
        
        # Prepare new version
        # Atomic swap of service endpoints
        # In real implementation, update service selectors atomically
        # For demo, simulate atomic swap
        await asyncio.sleep(2)
        
        logger.info(f"Atomic swap completed: {target.name}")
    
    async def _deploy_standard(self, deployment: ZeroDowntimeDeployment, 
                             target: DeploymentTarget):
        """Execute standard deployment"""
        logger.info(f"Executing standard deployment to {target.name}")
        
        # Standard deployment process
        # In real implementation, apply Kubernetes manifests
        # For demo, simulate deployment
        await asyncio.sleep(3)
        
        logger.info(f"Standard deployment completed: {target.name}")
    
    async def _phase_health_checks(self, deployment: ZeroDowntimeDeployment):
        """Health check phase"""
        logger.info(f"Phase: Health checks for {deployment.name}")
        deployment.current_phase = DeploymentPhase.HEALTH_CHECKING
        
        start_time = time.time()
        
        try:
            # Wait for startup grace period
            grace_period = self.config["health_checks"]["startup_grace_period"]
            logger.info(f"Waiting {grace_period}s for startup grace period...")
            await asyncio.sleep(grace_period)
            
            # Execute health checks
            health_results = await self._execute_health_checks(deployment)
            
            # Validate health check results
            if not all(result["healthy"] for result in health_results.values()):
                unhealthy_checks = [name for name, result in health_results.items() 
                                  if not result["healthy"]]
                raise Exception(f"Health checks failed: {', '.join(unhealthy_checks)}")
            
            duration = time.time() - start_time
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.HEALTH_CHECKING,
                message="All health checks passed",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"health_checks": health_results}
            )
            deployment.results.append(result)
            
        except Exception as e:
            duration = time.time() - start_time
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.HEALTH_CHECKING,
                message=f"Health checks failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"error": str(e)}
            )
            deployment.results.append(result)
            raise
    
    async def _execute_health_checks(self, deployment: ZeroDowntimeDeployment) -> Dict[str, Dict[str, Any]]:
        """Execute all health checks"""
        results = {}
        
        for health_check in deployment.health_checks:
            result = await self._execute_single_health_check(health_check)
            results[health_check.name] = result
        
        return results
    
    async def _execute_single_health_check(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Execute a single health check"""
        logger.info(f"Executing health check: {health_check.name}")
        
        for attempt in range(health_check.retries + 1):
            try:
                # In real implementation, make HTTP request to health endpoint
                # For demo, simulate health check
                await asyncio.sleep(1)
                
                # Simulate success most of the time
                import random
                if random.random() > 0.1:  # 90% success rate
                    return {
                        "healthy": True,
                        "status_code": health_check.expected_status,
                        "response_time_ms": 50,
                        "attempt": attempt + 1
                    }
                else:
                    if attempt < health_check.retries:
                        logger.warning(f"Health check {health_check.name} failed, retrying...")
                        await asyncio.sleep(health_check.interval_seconds)
                        continue
                    else:
                        return {
                            "healthy": False,
                            "status_code": 500,
                            "error": "Health check failed after retries",
                            "attempt": attempt + 1
                        }
                        
            except Exception as e:
                if attempt < health_check.retries:
                    logger.warning(f"Health check {health_check.name} error: {e}, retrying...")
                    await asyncio.sleep(health_check.interval_seconds)
                    continue
                else:
                    return {
                        "healthy": False,
                        "error": str(e),
                        "attempt": attempt + 1
                    }
        
        return {"healthy": False, "error": "Unknown error"}
    
    async def _phase_traffic_shifting(self, deployment: ZeroDowntimeDeployment):
        """Traffic shifting phase"""
        logger.info(f"Phase: Traffic shifting for {deployment.name}")
        deployment.current_phase = DeploymentPhase.TRAFFIC_SHIFTING
        
        start_time = time.time()
        
        try:
            strategy = self.config["traffic_shifting"]["strategy"]
            
            if strategy == "immediate":
                await self._shift_traffic_immediate(deployment)
            elif strategy == "gradual":
                await self._shift_traffic_gradual(deployment)
            else:
                await self._shift_traffic_weighted(deployment)
            
            duration = time.time() - start_time
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.TRAFFIC_SHIFTING,
                message="Traffic shifting completed",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"strategy": strategy}
            )
            deployment.results.append(result)
            
        except Exception as e:
            duration = time.time() - start_time
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.TRAFFIC_SHIFTING,
                message=f"Traffic shifting failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"error": str(e)}
            )
            deployment.results.append(result)
            raise
    
    async def _shift_traffic_immediate(self, deployment: ZeroDowntimeDeployment):
        """Immediate traffic shifting"""
        logger.info("Executing immediate traffic shift")
        
        # Update traffic rules to route 100% to new version
        await self._update_traffic_rules(deployment, 100)
        
        logger.info("Immediate traffic shift completed")
    
    async def _shift_traffic_gradual(self, deployment: ZeroDowntimeDeployment):
        """Gradual traffic shifting"""
        logger.info("Executing gradual traffic shift")
        
        steps = self.config["traffic_shifting"]["steps"]
        step_duration = self.config["traffic_shifting"]["step_duration_minutes"] * 60
        validation_duration = self.config["traffic_shifting"]["validation_duration_minutes"] * 60
        
        for step_percentage in steps:
            logger.info(f"Shifting {step_percentage}% traffic to new version")
            
            # Update traffic rules
            await self._update_traffic_rules(deployment, step_percentage)
            
            # Wait for step duration
            await asyncio.sleep(step_duration)
            
            # Validate metrics during this step
            await self._validate_traffic_step(deployment, step_percentage)
            
            # Additional validation period
            await asyncio.sleep(validation_duration)
        
        logger.info("Gradual traffic shift completed")
    
    async def _shift_traffic_weighted(self, deployment: ZeroDowntimeDeployment):
        """Weighted traffic shifting based on rules"""
        logger.info("Executing weighted traffic shift")
        
        for rule in deployment.traffic_rules:
            await self._apply_traffic_rule(rule)
        
        logger.info("Weighted traffic shift completed")
    
    async def _update_traffic_rules(self, deployment: ZeroDowntimeDeployment, percentage: int):
        """Update traffic routing rules"""
        if self.config["istio"]["enabled"]:
            # Update Istio VirtualService
            await self._update_istio_virtual_service(deployment, percentage)
        else:
            # Update Kubernetes service selectors
            await self._update_service_selectors(deployment, percentage)
    
    async def _update_istio_virtual_service(self, deployment: ZeroDowntimeDeployment, percentage: int):
        """Update Istio VirtualService for traffic splitting"""
        logger.info(f"Updating Istio VirtualService: {percentage}% to new version")
        
        # In real implementation, update VirtualService manifest
        # For demo, simulate update
        await asyncio.sleep(1)
        
        logger.info("Istio VirtualService updated")
    
    async def _update_service_selectors(self, deployment: ZeroDowntimeDeployment, percentage: int):
        """Update Kubernetes service selectors for traffic splitting"""
        logger.info(f"Updating service selectors: {percentage}% to new version")
        
        # In real implementation, update service selectors
        # For demo, simulate update
        await asyncio.sleep(1)
        
        logger.info("Service selectors updated")
    
    async def _apply_traffic_rule(self, rule: TrafficRule):
        """Apply a specific traffic rule"""
        logger.info(f"Applying traffic rule: {rule.name}")
        
        # In real implementation, apply rule to service mesh
        # For demo, simulate rule application
        await asyncio.sleep(0.5)
        
        logger.info(f"Traffic rule applied: {rule.name}")
    
    async def _validate_traffic_step(self, deployment: ZeroDowntimeDeployment, percentage: int):
        """Validate metrics during traffic shifting step"""
        logger.info(f"Validating traffic step: {percentage}%")
        
        # Check error rates, response times, etc.
        # In real implementation, query monitoring systems
        # For demo, simulate validation
        await asyncio.sleep(2)
        
        # Simulate occasional validation failure
        import random
        if random.random() < 0.05:  # 5% chance of failure
            raise Exception(f"Validation failed at {percentage}% traffic")
        
        logger.info(f"Traffic step validation passed: {percentage}%")
    
    async def _phase_validation(self, deployment: ZeroDowntimeDeployment):
        """Validation phase"""
        logger.info(f"Phase: Validation for {deployment.name}")
        deployment.current_phase = DeploymentPhase.VALIDATING
        
        start_time = time.time()
        
        try:
            # Validate deployment success
            await self._validate_deployment_success(deployment)
            
            # Validate performance metrics
            await self._validate_performance_metrics(deployment)
            
            # Validate business metrics
            await self._validate_business_metrics(deployment)
            
            duration = time.time() - start_time
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.VALIDATING,
                message="Validation completed successfully",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"all_validations_passed": True}
            )
            deployment.results.append(result)
            
        except Exception as e:
            duration = time.time() - start_time
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.VALIDATING,
                message=f"Validation failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"error": str(e)}
            )
            deployment.results.append(result)
            raise
    
    async def _validate_deployment_success(self, deployment: ZeroDowntimeDeployment):
        """Validate deployment was successful"""
        logger.info("Validating deployment success...")
        
        # Check pod status, readiness, etc.
        # In real implementation, query Kubernetes API
        # For demo, simulate validation
        await asyncio.sleep(2)
        
        logger.info("Deployment success validation passed")
    
    async def _validate_performance_metrics(self, deployment: ZeroDowntimeDeployment):
        """Validate performance metrics"""
        logger.info("Validating performance metrics...")
        
        # Check response times, throughput, error rates
        # In real implementation, query Prometheus
        # For demo, simulate validation
        await asyncio.sleep(3)
        
        logger.info("Performance metrics validation passed")
    
    async def _validate_business_metrics(self, deployment: ZeroDowntimeDeployment):
        """Validate business metrics"""
        logger.info("Validating business metrics...")
        
        # Check conversion rates, user engagement, etc.
        # In real implementation, query analytics systems
        # For demo, simulate validation
        await asyncio.sleep(2)
        
        logger.info("Business metrics validation passed")
    
    async def _phase_completion(self, deployment: ZeroDowntimeDeployment):
        """Completion phase"""
        logger.info(f"Phase: Completion for {deployment.name}")
        deployment.current_phase = DeploymentPhase.COMPLETING
        
        start_time = time.time()
        
        try:
            # Cleanup old versions
            await self._cleanup_old_versions(deployment)
            
            # Update feature flags
            await self._finalize_feature_flags(deployment)
            
            # Send notifications
            await self._send_completion_notifications(deployment)
            
            # Update deployment status
            deployment.current_phase = DeploymentPhase.COMPLETED
            deployment.completed_at = datetime.now()
            
            duration = time.time() - start_time
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.COMPLETING,
                message="Deployment completed successfully",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"cleanup_completed": True}
            )
            deployment.results.append(result)
            
            # Move to history
            if deployment.id in self.active_deployments:
                del self.active_deployments[deployment.id]
            self.deployment_history.append(deployment)
            
            logger.info(f"Zero-downtime deployment completed: {deployment.name}")
            
        except Exception as e:
            duration = time.time() - start_time
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.COMPLETING,
                message=f"Completion failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=duration,
                details={"error": str(e)}
            )
            deployment.results.append(result)
            logger.error(f"Completion phase failed: {e}")
    
    async def _cleanup_old_versions(self, deployment: ZeroDowntimeDeployment):
        """Cleanup old deployment versions"""
        logger.info("Cleaning up old versions...")
        
        # Remove old deployments, services, etc.
        # In real implementation, delete Kubernetes resources
        # For demo, simulate cleanup
        await asyncio.sleep(2)
        
        logger.info("Old versions cleanup completed")
    
    async def _finalize_feature_flags(self, deployment: ZeroDowntimeDeployment):
        """Finalize feature flag states"""
        if not self.config["feature_flags"]["enabled"]:
            return
        
        logger.info("Finalizing feature flags...")
        
        for flag in deployment.feature_flags:
            # Set flags to final state
            logger.info(f"Feature flag finalized: {flag.name}")
        
        logger.info("Feature flags finalization completed")
    
    async def _send_completion_notifications(self, deployment: ZeroDowntimeDeployment):
        """Send deployment completion notifications"""
        logger.info("Sending completion notifications...")
        
        # Send to Slack, email, etc.
        # In real implementation, send actual notifications
        # For demo, simulate notifications
        await asyncio.sleep(1)
        
        logger.info("Completion notifications sent")
    
    async def _handle_deployment_failure(self, deployment: ZeroDowntimeDeployment, error: str):
        """Handle deployment failure"""
        logger.error(f"Handling deployment failure: {error}")
        
        deployment.current_phase = DeploymentPhase.FAILED
        
        if self.config["rollback"]["automatic"]:
            await self._execute_rollback(deployment, error)
        else:
            logger.info("Automatic rollback disabled, manual intervention required")
    
    async def _execute_rollback(self, deployment: ZeroDowntimeDeployment, reason: str):
        """Execute deployment rollback"""
        logger.info(f"Executing rollback for {deployment.name}: {reason}")
        
        deployment.current_phase = DeploymentPhase.ROLLING_BACK
        
        try:
            # Rollback traffic routing
            await self._rollback_traffic_routing(deployment)
            
            # Rollback deployments
            await self._rollback_deployments(deployment)
            
            # Rollback database migrations
            await self._rollback_database_migrations(deployment)
            
            # Rollback feature flags
            await self._rollback_feature_flags(deployment)
            
            deployment.current_phase = DeploymentPhase.FAILED
            
            result = DeploymentResult(
                success=True,
                phase=DeploymentPhase.ROLLING_BACK,
                message=f"Rollback completed: {reason}",
                timestamp=datetime.now(),
                duration_seconds=0,
                details={"rollback_reason": reason},
                rollback_info={"completed": True}
            )
            deployment.results.append(result)
            
            logger.info(f"Rollback completed for {deployment.name}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            result = DeploymentResult(
                success=False,
                phase=DeploymentPhase.ROLLING_BACK,
                message=f"Rollback failed: {e}",
                timestamp=datetime.now(),
                duration_seconds=0,
                details={"error": str(e)}
            )
            deployment.results.append(result)
    
    async def _rollback_traffic_routing(self, deployment: ZeroDowntimeDeployment):
        """Rollback traffic routing"""
        logger.info("Rolling back traffic routing...")
        
        # Route traffic back to original version
        await self._update_traffic_rules(deployment, 0)  # 0% to new version
        
        logger.info("Traffic routing rollback completed")
    
    async def _rollback_deployments(self, deployment: ZeroDowntimeDeployment):
        """Rollback deployments"""
        logger.info("Rolling back deployments...")
        
        # Restore previous deployment versions
        # In real implementation, restore from backup
        await asyncio.sleep(3)
        
        logger.info("Deployment rollback completed")
    
    async def _rollback_database_migrations(self, deployment: ZeroDowntimeDeployment):
        """Rollback database migrations"""
        logger.info("Rolling back database migrations...")
        
        for migration in reversed(deployment.database_migrations):
            if migration.rollback_script_path:
                # Execute rollback script
                logger.info(f"Rolling back migration: {migration.name}")
                await asyncio.sleep(1)
        
        logger.info("Database migration rollback completed")
    
    async def _rollback_feature_flags(self, deployment: ZeroDowntimeDeployment):
        """Rollback feature flags"""
        if not self.config["feature_flags"]["enabled"]:
            return
        
        logger.info("Rolling back feature flags...")
        
        for flag in deployment.feature_flags:
            # Restore previous flag state
            logger.info(f"Rolling back feature flag: {flag.name}")
        
        logger.info("Feature flag rollback completed")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            return {
                "id": deployment.id,
                "name": deployment.name,
                "strategy": deployment.strategy.value,
                "current_phase": deployment.current_phase.value,
                "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
                "source_version": deployment.source_version,
                "target_version": deployment.target_version,
                "results": [asdict(result) for result in deployment.results]
            }
        
        # Check history
        for deployment in self.deployment_history:
            if deployment.id == deployment_id:
                return {
                    "id": deployment.id,
                    "name": deployment.name,
                    "strategy": deployment.strategy.value,
                    "current_phase": deployment.current_phase.value,
                    "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
                    "completed_at": deployment.completed_at.isoformat() if deployment.completed_at else None,
                    "source_version": deployment.source_version,
                    "target_version": deployment.target_version,
                    "results": [asdict(result) for result in deployment.results],
                    "completed": True
                }
        
        return None
    
    def get_deployment_report(self, deployment_id: str) -> Dict[str, Any]:
        """Generate deployment report"""
        status = self.get_deployment_status(deployment_id)
        if not status:
            return {"error": "Deployment not found"}
        
        # Calculate metrics
        total_duration = 0
        if status.get("completed_at") and status.get("started_at"):
            start = datetime.fromisoformat(status["started_at"])
            end = datetime.fromisoformat(status["completed_at"])
            total_duration = (end - start).total_seconds()
        
        successful_phases = len([r for r in status["results"] if r["success"]])
        total_phases = len(status["results"])
        
        report = {
            "deployment": status,
            "summary": {
                "total_duration_seconds": total_duration,
                "successful_phases": successful_phases,
                "total_phases": total_phases,
                "success_rate": successful_phases / total_phases if total_phases > 0 else 0,
                "current_status": status["current_phase"]
            },
            "phase_breakdown": {
                result["phase"]: {
                    "success": result["success"],
                    "duration_seconds": result["duration_seconds"],
                    "message": result["message"]
                }
                for result in status["results"]
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return report

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_zero_downtime_deployer():
        """Test the Zero-Downtime Deployer"""
        deployer = ZeroDowntimeDeployer()
        
        # Initialize
        await deployer.initialize()
        
        # Create test deployment
        deployment = ZeroDowntimeDeployment(
            id="zd-deploy-001",
            name="myapp",
            strategy=DeploymentStrategy.BLUE_GREEN,
            source_version="v1.0.0",
            target_version="v1.1.0",
            targets=[
                DeploymentTarget(
                    name="production",
                    namespace="default",
                    cluster="main",
                    region="us-west-2",
                    environment="prod",
                    replicas=3,
                    resources={"cpu": "500m", "memory": "512Mi"},
                    config_maps=[],
                    secrets=[],
                    volumes=[]
                )
            ],
            health_checks=[
                HealthCheck(
                    name="http-health",
                    endpoint="/health",
                    method="GET",
                    expected_status=200,
                    timeout_seconds=30,
                    interval_seconds=5,
                    retries=3,
                    headers={}
                )
            ],
            traffic_rules=[],
            feature_flags=[],
            database_migrations=[],
            rollback_config={},
            timeout_minutes=60,
            created_at=datetime.now()
        )
        
        # Start deployment
        deployment_id = await deployer.deploy(deployment)
        print(f"Started zero-downtime deployment: {deployment_id}")
        
        # Monitor progress
        for i in range(10):
            await asyncio.sleep(5)
            status = deployer.get_deployment_status(deployment_id)
            if status:
                print(f"Status: {status['current_phase']} - {len(status['results'])} phases completed")
                if status['current_phase'] in ['completed', 'failed']:
                    break
        
        # Get final report
        report = deployer.get_deployment_report(deployment_id)
        print(f"Deployment Report: {json.dumps(report, indent=2, default=str)}")
    
    # Run test
    asyncio.run(test_zero_downtime_deployer())