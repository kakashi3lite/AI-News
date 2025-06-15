#!/usr/bin/env python3
"""
Chaos Engineering Module - Resilience Testing for Deployments

This module implements chaos engineering experiments to validate system resilience
during canary deployments and production operations.

Features:
- Network chaos (latency, packet loss, partitions)
- Resource chaos (CPU, memory, disk stress)
- Pod chaos (kills, failures, restarts)
- Service mesh chaos (traffic manipulation)
- Database chaos (connection failures, slow queries)
- Integration with Kubernetes and monitoring systems

Author: Commander Solaris "DeployX" Vivante
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import yaml
import requests
import numpy as np

logger = logging.getLogger(__name__)

class ChaosType(Enum):
    """Types of chaos experiments"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_LOSS = "network_loss"
    NETWORK_PARTITION = "network_partition"
    POD_KILL = "pod_kill"
    POD_FAILURE = "pod_failure"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATABASE_SLOW = "database_slow"
    DATABASE_DISCONNECT = "database_disconnect"

class ExperimentStatus(Enum):
    """Status of chaos experiments"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ChaosExperiment:
    """Chaos experiment definition"""
    id: str
    name: str
    chaos_type: ChaosType
    target: str
    duration: int  # seconds
    intensity: float  # 0.0 to 1.0
    parameters: Dict[str, Any]
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ChaosEngineer:
    """
    Chaos Engineering system for testing deployment resilience
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[ChaosExperiment] = []
        self.prometheus_url = self.config.get("prometheus_url", "http://localhost:9090")
        self.kubernetes_enabled = self.config.get("kubernetes_enabled", False)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default chaos engineering configuration"""
        return {
            "prometheus_url": "http://localhost:9090",
            "kubernetes_enabled": False,
            "default_duration": 300,  # 5 minutes
            "safety_limits": {
                "max_concurrent_experiments": 3,
                "max_experiment_duration": 1800,  # 30 minutes
                "min_recovery_time": 60,  # 1 minute between experiments
                "production_intensity_limit": 0.3
            },
            "targets": {
                "canary_pods": ["app=myapp,version=canary"],
                "stable_pods": ["app=myapp,version=stable"],
                "database": ["app=postgres"],
                "cache": ["app=redis"]
            },
            "experiments": {
                "network_latency": {
                    "enabled": True,
                    "default_latency_ms": 100,
                    "max_latency_ms": 1000
                },
                "pod_kill": {
                    "enabled": True,
                    "kill_percentage": 0.1,
                    "grace_period": 30
                },
                "cpu_stress": {
                    "enabled": True,
                    "default_load": 0.5,
                    "max_load": 0.8
                }
            }
        }
    
    async def initialize(self):
        """Initialize the Chaos Engineer"""
        logger.info("Initializing Chaos Engineer...")
        
        try:
            # Test monitoring connectivity
            await self._test_monitoring_connection()
            
            # Validate configuration
            self._validate_config()
            
            # Initialize experiment templates
            self._initialize_experiment_templates()
            
            logger.info("Chaos Engineer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chaos Engineer: {e}")
            raise
    
    async def _test_monitoring_connection(self):
        """Test connection to monitoring systems"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                  params={"query": "up"}, timeout=10)
            response.raise_for_status()
            logger.info("Monitoring connection successful")
        except Exception as e:
            logger.warning(f"Monitoring connection failed: {e}")
    
    def _validate_config(self):
        """Validate chaos engineering configuration"""
        required_keys = ["safety_limits", "targets", "experiments"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate safety limits
        safety = self.config["safety_limits"]
        if safety["max_experiment_duration"] > 3600:  # 1 hour
            logger.warning("Max experiment duration exceeds 1 hour - this may be unsafe")
    
    def _initialize_experiment_templates(self):
        """Initialize predefined experiment templates"""
        self.experiment_templates = {
            "canary_network_test": {
                "name": "Canary Network Latency Test",
                "chaos_type": ChaosType.NETWORK_LATENCY,
                "target": "canary_pods",
                "duration": 300,
                "intensity": 0.3,
                "parameters": {"latency_ms": 100, "jitter_ms": 20}
            },
            "canary_pod_failure": {
                "name": "Canary Pod Failure Test",
                "chaos_type": ChaosType.POD_KILL,
                "target": "canary_pods",
                "duration": 180,
                "intensity": 0.2,
                "parameters": {"kill_percentage": 0.1}
            },
            "resource_stress_test": {
                "name": "Resource Stress Test",
                "chaos_type": ChaosType.CPU_STRESS,
                "target": "canary_pods",
                "duration": 240,
                "intensity": 0.5,
                "parameters": {"cpu_load": 0.7, "workers": 2}
            },
            "database_chaos": {
                "name": "Database Connection Chaos",
                "chaos_type": ChaosType.DATABASE_SLOW,
                "target": "database",
                "duration": 300,
                "intensity": 0.3,
                "parameters": {"delay_ms": 500, "error_rate": 0.05}
            }
        }
    
    async def create_experiment(self, template_name: str, 
                              custom_params: Optional[Dict] = None) -> str:
        """Create a new chaos experiment from template"""
        if template_name not in self.experiment_templates:
            raise ValueError(f"Unknown experiment template: {template_name}")
        
        template = self.experiment_templates[template_name].copy()
        
        # Apply custom parameters
        if custom_params:
            template.update(custom_params)
            if "parameters" in custom_params:
                template["parameters"].update(custom_params["parameters"])
        
        # Generate unique experiment ID
        experiment_id = f"chaos-{int(time.time())}-{random.randint(1000, 9999)}"
        
        # Create experiment object
        experiment = ChaosExperiment(
            id=experiment_id,
            name=template["name"],
            chaos_type=template["chaos_type"],
            target=template["target"],
            duration=template["duration"],
            intensity=template["intensity"],
            parameters=template["parameters"]
        )
        
        # Validate experiment safety
        self._validate_experiment_safety(experiment)
        
        # Store experiment
        self.active_experiments[experiment_id] = experiment
        
        logger.info(f"Created chaos experiment: {experiment_id} ({template_name})")
        return experiment_id
    
    def _validate_experiment_safety(self, experiment: ChaosExperiment):
        """Validate experiment safety constraints"""
        safety = self.config["safety_limits"]
        
        # Check concurrent experiments limit
        running_count = sum(1 for exp in self.active_experiments.values() 
                          if exp.status == ExperimentStatus.RUNNING)
        if running_count >= safety["max_concurrent_experiments"]:
            raise ValueError("Too many concurrent experiments running")
        
        # Check duration limit
        if experiment.duration > safety["max_experiment_duration"]:
            raise ValueError(f"Experiment duration exceeds limit: {experiment.duration}s")
        
        # Check intensity limit for production
        if experiment.intensity > safety["production_intensity_limit"]:
            logger.warning(f"High intensity experiment: {experiment.intensity}")
    
    async def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Start a chaos experiment"""
        if experiment_id not in self.active_experiments:
            return {"success": False, "error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.PENDING:
            return {"success": False, "error": "Experiment not in pending state"}
        
        logger.info(f"Starting chaos experiment: {experiment_id}")
        
        try:
            # Update experiment status
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_time = datetime.now()
            
            # Execute the chaos experiment
            await self._execute_experiment(experiment)
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "start_time": experiment.start_time.isoformat(),
                "duration": experiment.duration
            }
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.error = str(e)
            return {"success": False, "error": str(e)}
    
    async def _execute_experiment(self, experiment: ChaosExperiment):
        """Execute the actual chaos experiment"""
        logger.info(f"Executing {experiment.chaos_type.value} on {experiment.target}")
        
        try:
            # Collect baseline metrics
            baseline_metrics = await self._collect_baseline_metrics(experiment)
            
            # Apply chaos based on type
            if experiment.chaos_type == ChaosType.NETWORK_LATENCY:
                await self._apply_network_latency(experiment)
            elif experiment.chaos_type == ChaosType.POD_KILL:
                await self._apply_pod_kill(experiment)
            elif experiment.chaos_type == ChaosType.CPU_STRESS:
                await self._apply_cpu_stress(experiment)
            elif experiment.chaos_type == ChaosType.DATABASE_SLOW:
                await self._apply_database_chaos(experiment)
            else:
                await self._simulate_chaos(experiment)
            
            # Monitor during experiment
            await self._monitor_experiment(experiment)
            
            # Collect results
            experiment.results = await self._collect_experiment_results(
                experiment, baseline_metrics
            )
            
            # Clean up chaos
            await self._cleanup_experiment(experiment)
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_time = datetime.now()
            
            logger.info(f"Experiment {experiment.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment {experiment.id} failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.error = str(e)
            experiment.end_time = datetime.now()
            
            # Attempt cleanup even on failure
            try:
                await self._cleanup_experiment(experiment)
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed for {experiment.id}: {cleanup_error}")
    
    async def _collect_baseline_metrics(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Collect baseline metrics before starting chaos"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "response_time": await self._get_response_time_metric(),
            "error_rate": await self._get_error_rate_metric(),
            "throughput": await self._get_throughput_metric(),
            "cpu_usage": await self._get_cpu_usage_metric(),
            "memory_usage": await self._get_memory_usage_metric()
        }
        
        logger.info(f"Baseline metrics collected for {experiment.id}")
        return metrics
    
    async def _apply_network_latency(self, experiment: ChaosExperiment):
        """Apply network latency chaos"""
        params = experiment.parameters
        latency_ms = params.get("latency_ms", 100)
        jitter_ms = params.get("jitter_ms", 20)
        
        logger.info(f"Applying network latency: {latency_ms}ms ± {jitter_ms}ms")
        
        # In real implementation, this would use tools like tc (traffic control)
        # or Chaos Mesh to inject network latency
        # For demo, we'll simulate the effect
        await self._simulate_network_chaos(experiment, "latency", latency_ms)
    
    async def _apply_pod_kill(self, experiment: ChaosExperiment):
        """Apply pod kill chaos"""
        params = experiment.parameters
        kill_percentage = params.get("kill_percentage", 0.1)
        
        logger.info(f"Killing {kill_percentage*100}% of pods")
        
        # In real implementation, this would use kubectl or Kubernetes API
        # to kill pods matching the target selector
        await self._simulate_pod_chaos(experiment, "kill", kill_percentage)
    
    async def _apply_cpu_stress(self, experiment: ChaosExperiment):
        """Apply CPU stress chaos"""
        params = experiment.parameters
        cpu_load = params.get("cpu_load", 0.5)
        workers = params.get("workers", 2)
        
        logger.info(f"Applying CPU stress: {cpu_load*100}% load with {workers} workers")
        
        # In real implementation, this would use stress-ng or similar tools
        await self._simulate_resource_chaos(experiment, "cpu", cpu_load)
    
    async def _apply_database_chaos(self, experiment: ChaosExperiment):
        """Apply database chaos"""
        params = experiment.parameters
        delay_ms = params.get("delay_ms", 500)
        error_rate = params.get("error_rate", 0.05)
        
        logger.info(f"Applying database chaos: {delay_ms}ms delay, {error_rate*100}% error rate")
        
        # In real implementation, this would use database proxy or toxiproxy
        await self._simulate_database_chaos(experiment, delay_ms, error_rate)
    
    async def _simulate_chaos(self, experiment: ChaosExperiment):
        """Simulate chaos experiment for demo purposes"""
        logger.info(f"Simulating {experiment.chaos_type.value} chaos")
        
        # Simulate chaos application time
        await asyncio.sleep(2)
        
        # Store simulation parameters
        experiment.parameters["simulated"] = True
        experiment.parameters["simulation_start"] = datetime.now().isoformat()
    
    async def _simulate_network_chaos(self, experiment: ChaosExperiment, 
                                     chaos_type: str, intensity: float):
        """Simulate network chaos effects"""
        logger.info(f"Simulating network {chaos_type} with intensity {intensity}")
        await asyncio.sleep(1)
    
    async def _simulate_pod_chaos(self, experiment: ChaosExperiment, 
                                chaos_type: str, percentage: float):
        """Simulate pod chaos effects"""
        logger.info(f"Simulating pod {chaos_type} affecting {percentage*100}% of pods")
        await asyncio.sleep(1)
    
    async def _simulate_resource_chaos(self, experiment: ChaosExperiment, 
                                     resource_type: str, load: float):
        """Simulate resource chaos effects"""
        logger.info(f"Simulating {resource_type} stress with {load*100}% load")
        await asyncio.sleep(1)
    
    async def _simulate_database_chaos(self, experiment: ChaosExperiment, 
                                     delay_ms: int, error_rate: float):
        """Simulate database chaos effects"""
        logger.info(f"Simulating database chaos: {delay_ms}ms delay, {error_rate*100}% errors")
        await asyncio.sleep(1)
    
    async def _monitor_experiment(self, experiment: ChaosExperiment):
        """Monitor experiment progress and collect metrics"""
        logger.info(f"Monitoring experiment {experiment.id} for {experiment.duration}s")
        
        # Monitor in intervals
        monitor_interval = min(30, experiment.duration // 10)  # 10 samples max
        elapsed = 0
        
        while elapsed < experiment.duration:
            await asyncio.sleep(monitor_interval)
            elapsed += monitor_interval
            
            # Collect current metrics
            current_metrics = await self._collect_current_metrics()
            
            # Store metrics in experiment
            if "monitoring_data" not in experiment.parameters:
                experiment.parameters["monitoring_data"] = []
            
            experiment.parameters["monitoring_data"].append({
                "timestamp": datetime.now().isoformat(),
                "elapsed": elapsed,
                "metrics": current_metrics
            })
            
            logger.debug(f"Experiment {experiment.id} progress: {elapsed}/{experiment.duration}s")
    
    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        return {
            "response_time": await self._get_response_time_metric(),
            "error_rate": await self._get_error_rate_metric(),
            "throughput": await self._get_throughput_metric(),
            "cpu_usage": await self._get_cpu_usage_metric(),
            "memory_usage": await self._get_memory_usage_metric()
        }
    
    async def _get_response_time_metric(self) -> float:
        """Get response time metric from monitoring"""
        try:
            # In real implementation, query Prometheus for response time
            # For demo, simulate with some chaos effect
            base_time = 0.2  # 200ms baseline
            chaos_effect = random.uniform(0.8, 1.5)  # Chaos can increase response time
            return base_time * chaos_effect
        except Exception:
            return 0.2
    
    async def _get_error_rate_metric(self) -> float:
        """Get error rate metric from monitoring"""
        try:
            # Simulate error rate with chaos effect
            base_rate = 0.001  # 0.1% baseline
            chaos_effect = random.uniform(1.0, 3.0)  # Chaos can increase errors
            return min(0.1, base_rate * chaos_effect)
        except Exception:
            return 0.001
    
    async def _get_throughput_metric(self) -> float:
        """Get throughput metric from monitoring"""
        try:
            # Simulate throughput with chaos effect
            base_throughput = 1000  # 1000 RPS baseline
            chaos_effect = random.uniform(0.7, 1.0)  # Chaos can decrease throughput
            return base_throughput * chaos_effect
        except Exception:
            return 1000
    
    async def _get_cpu_usage_metric(self) -> float:
        """Get CPU usage metric from monitoring"""
        try:
            # Simulate CPU usage with chaos effect
            base_cpu = 0.4  # 40% baseline
            chaos_effect = random.uniform(1.0, 2.0)  # Chaos can increase CPU
            return min(1.0, base_cpu * chaos_effect)
        except Exception:
            return 0.4
    
    async def _get_memory_usage_metric(self) -> float:
        """Get memory usage metric from monitoring"""
        try:
            # Simulate memory usage with chaos effect
            base_memory = 0.6  # 60% baseline
            chaos_effect = random.uniform(1.0, 1.5)  # Chaos can increase memory
            return min(1.0, base_memory * chaos_effect)
        except Exception:
            return 0.6
    
    async def _collect_experiment_results(self, experiment: ChaosExperiment, 
                                        baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and analyze experiment results"""
        # Get final metrics
        final_metrics = await self._collect_current_metrics()
        
        # Calculate impact
        impact_analysis = self._analyze_chaos_impact(baseline_metrics, final_metrics)
        
        # Get monitoring data
        monitoring_data = experiment.parameters.get("monitoring_data", [])
        
        results = {
            "baseline_metrics": baseline_metrics,
            "final_metrics": final_metrics,
            "impact_analysis": impact_analysis,
            "monitoring_data": monitoring_data,
            "experiment_summary": {
                "duration": experiment.duration,
                "intensity": experiment.intensity,
                "chaos_type": experiment.chaos_type.value,
                "target": experiment.target
            },
            "resilience_score": self._calculate_resilience_score(impact_analysis),
            "recommendations": self._generate_resilience_recommendations(impact_analysis)
        }
        
        logger.info(f"Results collected for experiment {experiment.id}")
        return results
    
    def _analyze_chaos_impact(self, baseline: Dict[str, Any], 
                            final: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of chaos on system metrics"""
        impact = {}
        
        for metric in ["response_time", "error_rate", "throughput", "cpu_usage", "memory_usage"]:
            if metric in baseline and metric in final:
                baseline_val = baseline[metric]
                final_val = final[metric]
                
                if baseline_val > 0:
                    change_percent = ((final_val - baseline_val) / baseline_val) * 100
                else:
                    change_percent = 0
                
                impact[metric] = {
                    "baseline": baseline_val,
                    "final": final_val,
                    "change_percent": change_percent,
                    "degraded": self._is_metric_degraded(metric, change_percent)
                }
        
        return impact
    
    def _is_metric_degraded(self, metric: str, change_percent: float) -> bool:
        """Determine if a metric shows degradation"""
        # Define what constitutes degradation for each metric
        degradation_thresholds = {
            "response_time": 20,  # >20% increase is degradation
            "error_rate": 50,     # >50% increase is degradation
            "throughput": -10,    # >10% decrease is degradation
            "cpu_usage": 30,      # >30% increase is degradation
            "memory_usage": 25    # >25% increase is degradation
        }
        
        threshold = degradation_thresholds.get(metric, 20)
        
        if metric == "throughput":
            return change_percent < threshold  # Negative threshold for throughput
        else:
            return change_percent > threshold
    
    def _calculate_resilience_score(self, impact_analysis: Dict[str, Any]) -> float:
        """Calculate overall resilience score (0-100)"""
        scores = []
        weights = {
            "response_time": 0.25,
            "error_rate": 0.30,
            "throughput": 0.25,
            "cpu_usage": 0.10,
            "memory_usage": 0.10
        }
        
        for metric, weight in weights.items():
            if metric in impact_analysis:
                degraded = impact_analysis[metric]["degraded"]
                change_percent = abs(impact_analysis[metric]["change_percent"])
                
                # Score based on degradation and magnitude
                if not degraded:
                    metric_score = 100  # No degradation
                else:
                    # Score decreases with magnitude of change
                    metric_score = max(0, 100 - change_percent)
                
                scores.append(metric_score * weight)
        
        return sum(scores) if scores else 50  # Default to 50 if no metrics
    
    def _generate_resilience_recommendations(self, impact_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on chaos experiment results"""
        recommendations = []
        
        for metric, data in impact_analysis.items():
            if data["degraded"]:
                change = data["change_percent"]
                
                if metric == "response_time" and change > 50:
                    recommendations.append(
                        "High response time degradation detected - consider implementing circuit breakers"
                    )
                elif metric == "error_rate" and change > 100:
                    recommendations.append(
                        "Significant error rate increase - improve error handling and retry logic"
                    )
                elif metric == "throughput" and change < -20:
                    recommendations.append(
                        "Throughput significantly reduced - consider horizontal scaling strategies"
                    )
                elif metric == "cpu_usage" and change > 50:
                    recommendations.append(
                        "High CPU usage under chaos - optimize resource allocation"
                    )
                elif metric == "memory_usage" and change > 40:
                    recommendations.append(
                        "Memory usage spike detected - implement memory management improvements"
                    )
        
        if not recommendations:
            recommendations.append("System showed good resilience to chaos - continue monitoring")
        
        return recommendations
    
    async def _cleanup_experiment(self, experiment: ChaosExperiment):
        """Clean up chaos experiment effects"""
        logger.info(f"Cleaning up experiment {experiment.id}")
        
        try:
            # In real implementation, this would:
            # - Remove network policies
            # - Stop stress processes
            # - Restore normal database connections
            # - Remove any temporary chaos resources
            
            # For demo, simulate cleanup
            await asyncio.sleep(1)
            
            experiment.parameters["cleanup_completed"] = True
            experiment.parameters["cleanup_time"] = datetime.now().isoformat()
            
            logger.info(f"Cleanup completed for experiment {experiment.id}")
            
        except Exception as e:
            logger.error(f"Cleanup failed for experiment {experiment.id}: {e}")
            raise
    
    async def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Stop a running chaos experiment"""
        if experiment_id not in self.active_experiments:
            return {"success": False, "error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            return {"success": False, "error": "Experiment not running"}
        
        logger.info(f"Stopping chaos experiment: {experiment_id}")
        
        try:
            # Clean up experiment
            await self._cleanup_experiment(experiment)
            
            # Update status
            experiment.status = ExperimentStatus.CANCELLED
            experiment.end_time = datetime.now()
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "end_time": experiment.end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of a chaos experiment"""
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        
        status = {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.status.value,
            "chaos_type": experiment.chaos_type.value,
            "target": experiment.target,
            "duration": experiment.duration,
            "intensity": experiment.intensity
        }
        
        if experiment.start_time:
            status["start_time"] = experiment.start_time.isoformat()
            
            if experiment.status == ExperimentStatus.RUNNING:
                elapsed = (datetime.now() - experiment.start_time).total_seconds()
                status["elapsed"] = int(elapsed)
                status["remaining"] = max(0, experiment.duration - int(elapsed))
        
        if experiment.end_time:
            status["end_time"] = experiment.end_time.isoformat()
        
        if experiment.error:
            status["error"] = experiment.error
        
        if experiment.results:
            status["resilience_score"] = experiment.results.get("resilience_score", 0)
        
        return status
    
    def list_experiments(self, status_filter: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List chaos experiments"""
        experiments = []
        
        for experiment in self.active_experiments.values():
            if status_filter is None or experiment.status == status_filter:
                experiments.append(self.get_experiment_status(experiment.id))
        
        return experiments
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed results of a completed experiment"""
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        
        if experiment.status not in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
            return {"error": "Experiment not completed"}
        
        return {
            "experiment": asdict(experiment),
            "results": experiment.results or {},
            "export_time": datetime.now().isoformat()
        }
    
    async def run_canary_chaos_suite(self, canary_target: str) -> Dict[str, Any]:
        """Run a comprehensive chaos test suite for canary deployment"""
        logger.info(f"Running canary chaos suite for target: {canary_target}")
        
        suite_results = {
            "suite_id": f"canary-chaos-{int(time.time())}",
            "target": canary_target,
            "start_time": datetime.now().isoformat(),
            "experiments": [],
            "overall_resilience_score": 0.0,
            "recommendations": []
        }
        
        # Define chaos test sequence
        test_sequence = [
            ("canary_network_test", {"target": canary_target, "duration": 180}),
            ("canary_pod_failure", {"target": canary_target, "duration": 120}),
            ("resource_stress_test", {"target": canary_target, "duration": 240})
        ]
        
        resilience_scores = []
        
        try:
            for template_name, custom_params in test_sequence:
                logger.info(f"Running chaos test: {template_name}")
                
                # Create and start experiment
                exp_id = await self.create_experiment(template_name, custom_params)
                result = await self.start_experiment(exp_id)
                
                if result["success"]:
                    # Wait for completion
                    while True:
                        status = self.get_experiment_status(exp_id)
                        if status["status"] in ["completed", "failed", "cancelled"]:
                            break
                        await asyncio.sleep(10)
                    
                    # Get results
                    exp_results = self.get_experiment_results(exp_id)
                    suite_results["experiments"].append(exp_results)
                    
                    if "results" in exp_results and "resilience_score" in exp_results["results"]:
                        resilience_scores.append(exp_results["results"]["resilience_score"])
                    
                    # Wait between experiments
                    await asyncio.sleep(60)
                
                else:
                    logger.error(f"Failed to start experiment {template_name}: {result['error']}")
            
            # Calculate overall resilience score
            if resilience_scores:
                suite_results["overall_resilience_score"] = sum(resilience_scores) / len(resilience_scores)
            
            # Generate overall recommendations
            suite_results["recommendations"] = self._generate_suite_recommendations(suite_results)
            
            suite_results["end_time"] = datetime.now().isoformat()
            suite_results["success"] = True
            
            logger.info(f"Canary chaos suite completed with resilience score: {suite_results['overall_resilience_score']:.1f}")
            
        except Exception as e:
            logger.error(f"Canary chaos suite failed: {e}")
            suite_results["success"] = False
            suite_results["error"] = str(e)
            suite_results["end_time"] = datetime.now().isoformat()
        
        return suite_results
    
    def _generate_suite_recommendations(self, suite_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on chaos suite results"""
        recommendations = []
        score = suite_results["overall_resilience_score"]
        
        if score >= 80:
            recommendations.append("Excellent resilience - canary is ready for production")
        elif score >= 60:
            recommendations.append("Good resilience - minor improvements recommended")
        elif score >= 40:
            recommendations.append("Moderate resilience - address identified issues before promotion")
        else:
            recommendations.append("Poor resilience - significant improvements required")
        
        # Aggregate recommendations from individual experiments
        all_recommendations = set()
        for exp in suite_results["experiments"]:
            if "results" in exp and "recommendations" in exp["results"]:
                all_recommendations.update(exp["results"]["recommendations"])
        
        recommendations.extend(list(all_recommendations))
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_chaos_engineer():
        """Test the Chaos Engineer"""
        engineer = ChaosEngineer()
        
        # Initialize
        await engineer.initialize()
        
        # Create and run a simple experiment
        exp_id = await engineer.create_experiment("canary_network_test")
        print(f"Created experiment: {exp_id}")
        
        # Start experiment
        result = await engineer.start_experiment(exp_id)
        print(f"Started experiment: {result}")
        
        # Monitor progress
        while True:
            status = engineer.get_experiment_status(exp_id)
            print(f"Status: {status['status']} - {status.get('elapsed', 0)}/{status.get('duration', 0)}s")
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                break
            
            await asyncio.sleep(5)
        
        # Get results
        results = engineer.get_experiment_results(exp_id)
        print(f"Results: Resilience Score = {results['results']['resilience_score']:.1f}")
        print(f"Recommendations: {results['results']['recommendations']}")
    
    # Run test
    asyncio.run(test_chaos_engineer())