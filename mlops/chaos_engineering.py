#!/usr/bin/env python3
"""
Chaos Engineering Framework
Dr. Aurora "CodeForge" Synth's Chaos-Driven Resilience Testing

Integrates chaos experiments into CI/CD pipelines to validate system resilience
and self-healing capabilities under various failure conditions.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor

import yaml
import psutil
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

class ExperimentType(Enum):
    """Types of chaos experiments"""
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_IO_STRESS = "disk_io_stress"
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    SERVICE_KILL = "service_kill"
    DATABASE_FAILURE = "database_failure"
    API_LATENCY = "api_latency"
    API_ERROR_INJECTION = "api_error_injection"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_DRIFT = "configuration_drift"
    SECURITY_BREACH = "security_breach"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class ExperimentStatus(Enum):
    """Status of chaos experiments"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"
    TIMEOUT = "timeout"

class BlastRadius(Enum):
    """Blast radius for chaos experiments"""
    MINIMAL = "minimal"  # Single component
    LIMITED = "limited"  # Single service
    MODERATE = "moderate"  # Multiple services
    EXTENSIVE = "extensive"  # Cross-system
    MAXIMUM = "maximum"  # Full system

@dataclass
class ExperimentConfig:
    """Configuration for a chaos experiment"""
    name: str
    experiment_type: ExperimentType
    duration: int  # seconds
    intensity: float = 0.5  # 0.0 to 1.0
    blast_radius: BlastRadius = BlastRadius.LIMITED
    target_components: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    rollback_strategy: str = "automatic"
    safety_checks: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class ExperimentResult:
    """Result of a chaos experiment"""
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    error_message: str = ""
    recovery_time: float = 0.0
    resilience_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ResilienceMetrics:
    """System resilience metrics"""
    availability: float = 0.0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    recovery_time: float = 0.0
    throughput: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    slo_compliance: float = 0.0
    mttr: float = 0.0  # Mean Time To Recovery
    mtbf: float = 0.0  # Mean Time Between Failures

class SystemMonitor:
    """Monitors system health during chaos experiments"""
    
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger("system_monitor")
        self.baseline_metrics = None
        self.monitoring_active = False
        self.metrics_history = []
        
    async def start_monitoring(self) -> None:
        """Start system monitoring"""
        self.monitoring_active = True
        self.baseline_metrics = await self.collect_metrics()
        self.logger.info("System monitoring started")
        
        # Start background monitoring task
        asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> ResilienceMetrics:
        """Stop monitoring and return final metrics"""
        self.monitoring_active = False
        final_metrics = await self.collect_metrics()
        
        # Calculate resilience metrics
        resilience = self._calculate_resilience_metrics(final_metrics)
        
        self.logger.info("System monitoring stopped")
        return resilience
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
                await asyncio.sleep(5)  # Collect metrics every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free': disk.free,
                    'network_bytes_sent': network.bytes_sent,
                    'network_bytes_recv': network.bytes_recv,
                    'process_count': process_count
                },
                'application': await self._collect_app_metrics()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    async def _collect_app_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        app_metrics = {
            'response_time': 0.0,
            'error_rate': 0.0,
            'throughput': 0.0,
            'active_connections': 0
        }
        
        try:
            # Try to collect metrics from common endpoints
            endpoints = [
                'http://localhost:8000/health',
                'http://localhost:3000/health',
                'http://localhost:5000/health'
            ]
            
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    response = requests.get(endpoint, timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        app_metrics['response_time'] = response_time
                        app_metrics['error_rate'] = 0.0
                        
                        # Try to parse metrics from response
                        try:
                            data = response.json()
                            if 'metrics' in data:
                                app_metrics.update(data['metrics'])
                        except:
                            pass
                        
                        break
                    else:
                        app_metrics['error_rate'] = 100.0
                        
                except requests.RequestException:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Failed to collect app metrics: {e}")
        
        return app_metrics
    
    def _calculate_resilience_metrics(self, final_metrics: Dict[str, Any]) -> ResilienceMetrics:
        """Calculate resilience metrics from monitoring data"""
        if not self.metrics_history or not self.baseline_metrics:
            return ResilienceMetrics()
        
        try:
            # Calculate availability
            successful_checks = len([m for m in self.metrics_history 
                                   if m['metrics'].get('application', {}).get('error_rate', 100) < 50])
            availability = (successful_checks / len(self.metrics_history)) * 100
            
            # Calculate response time percentiles
            response_times = [m['metrics'].get('application', {}).get('response_time', 0) 
                            for m in self.metrics_history]
            response_times = [rt for rt in response_times if rt > 0]
            
            if response_times:
                response_times.sort()
                p95_index = int(len(response_times) * 0.95)
                response_time_p95 = response_times[p95_index] if p95_index < len(response_times) else response_times[-1]
            else:
                response_time_p95 = 0.0
            
            # Calculate error rate
            error_rates = [m['metrics'].get('application', {}).get('error_rate', 0) 
                         for m in self.metrics_history]
            avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0.0
            
            # Calculate resource utilization
            cpu_values = [m['metrics'].get('system', {}).get('cpu_percent', 0) 
                        for m in self.metrics_history]
            memory_values = [m['metrics'].get('system', {}).get('memory_percent', 0) 
                           for m in self.metrics_history]
            
            avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
            avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0.0
            
            # Calculate SLO compliance (assuming 99% availability, <500ms response time)
            slo_violations = 0
            for m in self.metrics_history:
                app_metrics = m['metrics'].get('application', {})
                if (app_metrics.get('error_rate', 0) > 1.0 or 
                    app_metrics.get('response_time', 0) > 500):
                    slo_violations += 1
            
            slo_compliance = ((len(self.metrics_history) - slo_violations) / len(self.metrics_history)) * 100
            
            return ResilienceMetrics(
                availability=availability,
                response_time_p95=response_time_p95,
                error_rate=avg_error_rate,
                recovery_time=0.0,  # Will be calculated by experiment
                throughput=final_metrics.get('application', {}).get('throughput', 0),
                resource_utilization={
                    'cpu': avg_cpu,
                    'memory': avg_memory
                },
                slo_compliance=slo_compliance,
                mttr=0.0,  # Will be calculated across experiments
                mtbf=0.0   # Will be calculated across experiments
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate resilience metrics: {e}")
            return ResilienceMetrics()

class ChaosExperiment:
    """Base class for chaos experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger(f"chaos_experiment_{config.name}")
        self.monitor = SystemMonitor()
        self.experiment_id = f"{config.name}_{int(time.time())}"
        
    async def execute(self) -> ExperimentResult:
        """Execute the chaos experiment"""
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            config=self.config,
            status=ExperimentStatus.PENDING,
            start_time=datetime.now()
        )
        
        try:
            self.logger.info(f"Starting chaos experiment: {self.config.name}")
            
            # Pre-experiment safety checks
            if not await self._safety_checks():
                result.status = ExperimentStatus.ABORTED
                result.error_message = "Safety checks failed"
                return result
            
            # Start monitoring
            await self.monitor.start_monitoring()
            
            # Execute experiment phases
            result.status = ExperimentStatus.RUNNING
            
            # Inject chaos
            await self._inject_chaos()
            
            # Wait for experiment duration
            await self._monitor_experiment()
            
            # Recovery phase
            recovery_start = time.time()
            await self._recover()
            result.recovery_time = time.time() - recovery_start
            
            # Stop monitoring and collect final metrics
            resilience_metrics = await self.monitor.stop_monitoring()
            
            # Evaluate results
            result = await self._evaluate_results(result, resilience_metrics)
            
            result.status = ExperimentStatus.SUCCESS
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
            self.logger.info(f"Chaos experiment completed: {self.config.name}")
            
        except asyncio.TimeoutError:
            result.status = ExperimentStatus.TIMEOUT
            result.error_message = "Experiment timed out"
            await self._emergency_recovery()
            
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"Chaos experiment failed: {e}")
            await self._emergency_recovery()
        
        finally:
            if result.end_time is None:
                result.end_time = datetime.now()
                result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    async def _safety_checks(self) -> bool:
        """Perform pre-experiment safety checks"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Ensure system is not already under stress
            if cpu_percent > 80:
                self.logger.warning(f"High CPU usage detected: {cpu_percent}%")
                return False
            
            if memory.percent > 85:
                self.logger.warning(f"High memory usage detected: {memory.percent}%")
                return False
            
            if (disk.used / disk.total) > 0.9:
                self.logger.warning(f"High disk usage detected: {(disk.used / disk.total) * 100:.1f}%")
                return False
            
            # Check if critical services are running
            for check in self.config.safety_checks:
                if not await self._execute_safety_check(check):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False
    
    async def _execute_safety_check(self, check: str) -> bool:
        """Execute a specific safety check"""
        try:
            if check.startswith('http://'):
                # HTTP health check
                response = requests.get(check, timeout=10)
                return response.status_code == 200
            elif check.startswith('process:'):
                # Process check
                process_name = check.split(':', 1)[1]
                for proc in psutil.process_iter(['name']):
                    if proc.info['name'] == process_name:
                        return True
                return False
            elif check.startswith('port:'):
                # Port check
                port = int(check.split(':', 1)[1])
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                return result == 0
            else:
                self.logger.warning(f"Unknown safety check type: {check}")
                return True
                
        except Exception as e:
            self.logger.error(f"Safety check '{check}' failed: {e}")
            return False
    
    async def _inject_chaos(self):
        """Inject chaos based on experiment type"""
        experiment_methods = {
            ExperimentType.CPU_STRESS: self._cpu_stress,
            ExperimentType.MEMORY_PRESSURE: self._memory_pressure,
            ExperimentType.DISK_IO_STRESS: self._disk_io_stress,
            ExperimentType.NETWORK_LATENCY: self._network_latency,
            ExperimentType.NETWORK_PARTITION: self._network_partition,
            ExperimentType.NETWORK_PACKET_LOSS: self._packet_loss,
            ExperimentType.SERVICE_KILL: self._service_kill,
            ExperimentType.API_LATENCY: self._api_latency,
            ExperimentType.API_ERROR_INJECTION: self._api_error_injection,
            ExperimentType.DEPENDENCY_FAILURE: self._dependency_failure
        }
        
        method = experiment_methods.get(self.config.experiment_type)
        if method:
            await method()
        else:
            raise NotImplementedError(f"Experiment type {self.config.experiment_type} not implemented")
    
    async def _monitor_experiment(self):
        """Monitor the experiment during execution"""
        await asyncio.sleep(self.config.duration)
    
    async def _recover(self):
        """Recover from chaos injection"""
        # Default recovery - stop any ongoing chaos processes
        # Specific experiments should override this method
        pass
    
    async def _emergency_recovery(self):
        """Emergency recovery in case of experiment failure"""
        try:
            await self._recover()
            self.logger.info("Emergency recovery completed")
        except Exception as e:
            self.logger.error(f"Emergency recovery failed: {e}")
    
    async def _evaluate_results(self, result: ExperimentResult, 
                              resilience_metrics: ResilienceMetrics) -> ExperimentResult:
        """Evaluate experiment results and generate recommendations"""
        result.metrics = {
            'availability': resilience_metrics.availability,
            'response_time_p95': resilience_metrics.response_time_p95,
            'error_rate': resilience_metrics.error_rate,
            'slo_compliance': resilience_metrics.slo_compliance,
            'resource_utilization': resilience_metrics.resource_utilization
        }
        
        # Calculate resilience score (0-100)
        score_components = [
            min(resilience_metrics.availability, 100) * 0.3,  # 30% weight
            max(0, 100 - resilience_metrics.error_rate) * 0.25,  # 25% weight
            min(resilience_metrics.slo_compliance, 100) * 0.25,  # 25% weight
            max(0, 100 - (result.recovery_time / 60) * 10) * 0.2  # 20% weight (recovery time in minutes)
        ]
        
        result.resilience_score = sum(score_components)
        
        # Generate recommendations
        recommendations = []
        
        if resilience_metrics.availability < 99:
            recommendations.append(f"Improve availability from {resilience_metrics.availability:.1f}% to 99%+")
        
        if resilience_metrics.response_time_p95 > 500:
            recommendations.append(f"Optimize response time from {resilience_metrics.response_time_p95:.0f}ms to <500ms")
        
        if resilience_metrics.error_rate > 1:
            recommendations.append(f"Reduce error rate from {resilience_metrics.error_rate:.1f}% to <1%")
        
        if result.recovery_time > 60:
            recommendations.append(f"Improve recovery time from {result.recovery_time:.1f}s to <60s")
        
        cpu_util = resilience_metrics.resource_utilization.get('cpu', 0)
        if cpu_util > 80:
            recommendations.append(f"Optimize CPU usage from {cpu_util:.1f}% during stress")
        
        memory_util = resilience_metrics.resource_utilization.get('memory', 0)
        if memory_util > 85:
            recommendations.append(f"Optimize memory usage from {memory_util:.1f}% during stress")
        
        result.recommendations = recommendations
        
        # Add observations
        result.observations = [
            f"System maintained {resilience_metrics.availability:.1f}% availability",
            f"95th percentile response time: {resilience_metrics.response_time_p95:.0f}ms",
            f"Average error rate: {resilience_metrics.error_rate:.1f}%",
            f"Recovery completed in {result.recovery_time:.1f} seconds",
            f"Overall resilience score: {result.resilience_score:.1f}/100"
        ]
        
        return result
    
    # Specific chaos injection methods
    async def _cpu_stress(self):
        """Inject CPU stress"""
        intensity = self.config.intensity
        duration = self.config.duration
        
        # Create CPU stress using busy loops
        def cpu_stress_worker():
            end_time = time.time() + duration
            while time.time() < end_time:
                # Busy loop with some sleep to control intensity
                for _ in range(int(1000000 * intensity)):
                    pass
                time.sleep(0.01 * (1 - intensity))
        
        # Start stress workers based on intensity
        num_workers = max(1, int(psutil.cpu_count() * intensity))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(cpu_stress_worker) for _ in range(num_workers)]
            
            # Wait for completion
            await asyncio.sleep(duration)
    
    async def _memory_pressure(self):
        """Inject memory pressure"""
        intensity = self.config.intensity
        duration = self.config.duration
        
        # Calculate memory to allocate
        available_memory = psutil.virtual_memory().available
        memory_to_allocate = int(available_memory * intensity * 0.8)  # 80% of target
        
        # Allocate memory in chunks
        memory_chunks = []
        chunk_size = 1024 * 1024 * 100  # 100MB chunks
        
        try:
            while len(memory_chunks) * chunk_size < memory_to_allocate:
                chunk = bytearray(chunk_size)
                # Write to memory to ensure it's actually allocated
                for i in range(0, chunk_size, 4096):
                    chunk[i] = 1
                memory_chunks.append(chunk)
                await asyncio.sleep(0.1)  # Small delay to avoid overwhelming system
            
            # Hold memory for duration
            await asyncio.sleep(duration)
            
        finally:
            # Release memory
            memory_chunks.clear()
    
    async def _disk_io_stress(self):
        """Inject disk I/O stress"""
        intensity = self.config.intensity
        duration = self.config.duration
        
        # Create temporary files for I/O stress
        temp_files = []
        file_size = int(1024 * 1024 * 10 * intensity)  # 10MB * intensity
        
        def io_worker(file_path):
            end_time = time.time() + duration
            while time.time() < end_time:
                try:
                    # Write data
                    with open(file_path, 'wb') as f:
                        f.write(b'0' * file_size)
                    
                    # Read data
                    with open(file_path, 'rb') as f:
                        f.read()
                    
                    time.sleep(0.1 * (1 - intensity))
                    
                except Exception as e:
                    self.logger.warning(f"I/O stress error: {e}")
                    break
        
        try:
            # Create multiple I/O workers
            num_workers = max(1, int(4 * intensity))
            
            for i in range(num_workers):
                temp_file = f"/tmp/chaos_io_stress_{self.experiment_id}_{i}.tmp"
                temp_files.append(temp_file)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(io_worker, temp_file) for temp_file in temp_files]
                
                # Wait for completion
                await asyncio.sleep(duration)
                
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {temp_file}: {e}")
    
    async def _network_latency(self):
        """Inject network latency"""
        # This would typically use tools like tc (traffic control) on Linux
        # For simulation, we'll just log the action
        latency_ms = int(self.config.intensity * 1000)  # Convert to milliseconds
        
        self.logger.info(f"Simulating {latency_ms}ms network latency for {self.config.duration}s")
        
        # In a real implementation, you would use:
        # tc qdisc add dev eth0 root netem delay {latency_ms}ms
        
        await asyncio.sleep(self.config.duration)
        
        # Recovery would be:
        # tc qdisc del dev eth0 root
    
    async def _network_partition(self):
        """Simulate network partition"""
        targets = self.config.target_components or ['external_api']
        
        self.logger.info(f"Simulating network partition to {targets} for {self.config.duration}s")
        
        # In a real implementation, you would block network access to specific hosts/ports
        # using iptables or similar tools
        
        await asyncio.sleep(self.config.duration)
    
    async def _packet_loss(self):
        """Inject packet loss"""
        loss_percent = int(self.config.intensity * 100)
        
        self.logger.info(f"Simulating {loss_percent}% packet loss for {self.config.duration}s")
        
        # In a real implementation:
        # tc qdisc add dev eth0 root netem loss {loss_percent}%
        
        await asyncio.sleep(self.config.duration)
    
    async def _service_kill(self):
        """Kill and restart services"""
        targets = self.config.target_components or ['web_server']
        
        for target in targets:
            self.logger.info(f"Killing service: {target}")
            
            # Find and kill process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if target.lower() in proc.info['name'].lower():
                    try:
                        proc.terminate()
                        await asyncio.sleep(2)
                        if proc.is_running():
                            proc.kill()
                        self.logger.info(f"Killed process {proc.info['name']} (PID: {proc.info['pid']})")
                        break
                    except Exception as e:
                        self.logger.warning(f"Failed to kill process {proc.info['name']}: {e}")
        
        # Wait for recovery mechanisms to restart services
        await asyncio.sleep(self.config.duration)
    
    async def _api_latency(self):
        """Inject API latency"""
        # This would typically be implemented as middleware or proxy
        latency_ms = int(self.config.intensity * 2000)  # Up to 2 seconds
        
        self.logger.info(f"Simulating {latency_ms}ms API latency for {self.config.duration}s")
        
        await asyncio.sleep(self.config.duration)
    
    async def _api_error_injection(self):
        """Inject API errors"""
        error_rate = int(self.config.intensity * 100)
        
        self.logger.info(f"Simulating {error_rate}% API error rate for {self.config.duration}s")
        
        await asyncio.sleep(self.config.duration)
    
    async def _dependency_failure(self):
        """Simulate dependency failures"""
        dependencies = self.config.target_components or ['database', 'cache', 'external_api']
        
        for dep in dependencies:
            self.logger.info(f"Simulating failure of dependency: {dep}")
        
        await asyncio.sleep(self.config.duration)

class ChaosOrchestrator:
    """Orchestrates multiple chaos experiments"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.console = Console()
        self.logger = logging.getLogger("chaos_orchestrator")
        self.experiments: List[ExperimentConfig] = []
        self.results: List[ExperimentResult] = []
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load experiment configurations from file"""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self.experiments = []
            for exp_data in config_data.get('experiments', []):
                config = ExperimentConfig(
                    name=exp_data['name'],
                    experiment_type=ExperimentType(exp_data['type']),
                    duration=exp_data.get('duration', 60),
                    intensity=exp_data.get('intensity', 0.5),
                    blast_radius=BlastRadius(exp_data.get('blast_radius', 'limited')),
                    target_components=exp_data.get('target_components', []),
                    parameters=exp_data.get('parameters', {}),
                    success_criteria=exp_data.get('success_criteria', {}),
                    rollback_strategy=exp_data.get('rollback_strategy', 'automatic'),
                    safety_checks=exp_data.get('safety_checks', []),
                    tags=exp_data.get('tags', [])
                )
                self.experiments.append(config)
                
            self.logger.info(f"Loaded {len(self.experiments)} experiment configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
            raise
    
    def add_experiment(self, config: ExperimentConfig):
        """Add an experiment configuration"""
        self.experiments.append(config)
    
    async def run_experiments(self, parallel: bool = False, 
                            tags: Optional[List[str]] = None) -> List[ExperimentResult]:
        """Run chaos experiments"""
        # Filter experiments by tags if specified
        experiments_to_run = self.experiments
        if tags:
            experiments_to_run = [
                exp for exp in self.experiments 
                if any(tag in exp.tags for tag in tags)
            ]
        
        if not experiments_to_run:
            self.logger.warning("No experiments to run")
            return []
        
        self.console.print(Panel.fit(
            f"[bold red]🔥 Chaos Engineering Suite[/bold red]\n"
            f"[dim]Running {len(experiments_to_run)} experiments[/dim]",
            border_style="red"
        ))
        
        if parallel:
            results = await self._run_parallel(experiments_to_run)
        else:
            results = await self._run_sequential(experiments_to_run)
        
        self.results.extend(results)
        
        # Generate summary report
        await self._generate_summary_report(results)
        
        return results
    
    async def _run_sequential(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run experiments sequentially"""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task("Chaos Experiments", total=len(experiments))
            
            for exp_config in experiments:
                exp_task = progress.add_task(f"Running {exp_config.name}...", total=None)
                
                try:
                    experiment = ChaosExperiment(exp_config)
                    result = await experiment.execute()
                    results.append(result)
                    
                    # Update progress
                    status_emoji = "✅" if result.status == ExperimentStatus.SUCCESS else "❌"
                    progress.update(
                        exp_task, 
                        description=f"{status_emoji} {exp_config.name} ({result.duration:.1f}s)", 
                        completed=True
                    )
                    
                except Exception as e:
                    self.logger.error(f"Experiment {exp_config.name} failed: {e}")
                    result = ExperimentResult(
                        experiment_id=f"{exp_config.name}_failed",
                        config=exp_config,
                        status=ExperimentStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(e)
                    )
                    results.append(result)
                
                progress.advance(main_task)
                
                # Wait between experiments
                await asyncio.sleep(5)
        
        return results
    
    async def _run_parallel(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run experiments in parallel (with caution)"""
        # Group experiments by blast radius to avoid conflicts
        minimal_experiments = [e for e in experiments if e.blast_radius == BlastRadius.MINIMAL]
        other_experiments = [e for e in experiments if e.blast_radius != BlastRadius.MINIMAL]
        
        results = []
        
        # Run minimal blast radius experiments in parallel
        if minimal_experiments:
            tasks = []
            for exp_config in minimal_experiments:
                experiment = ChaosExperiment(exp_config)
                task = asyncio.create_task(experiment.execute())
                tasks.append(task)
            
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    error_result = ExperimentResult(
                        experiment_id=f"{minimal_experiments[i].name}_failed",
                        config=minimal_experiments[i],
                        status=ExperimentStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        # Run other experiments sequentially
        if other_experiments:
            sequential_results = await self._run_sequential(other_experiments)
            results.extend(sequential_results)
        
        return results
    
    async def _generate_summary_report(self, results: List[ExperimentResult]):
        """Generate a summary report of all experiments"""
        # Create summary table
        table = Table(title="Chaos Engineering Results", show_header=True, header_style="bold magenta")
        table.add_column("Experiment", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Resilience Score", style="red")
        table.add_column("Recovery Time", style="orange")
        
        total_experiments = len(results)
        successful_experiments = len([r for r in results if r.status == ExperimentStatus.SUCCESS])
        failed_experiments = len([r for r in results if r.status == ExperimentStatus.FAILED])
        
        avg_resilience_score = 0
        avg_recovery_time = 0
        
        if results:
            resilience_scores = [r.resilience_score for r in results if r.resilience_score > 0]
            recovery_times = [r.recovery_time for r in results if r.recovery_time > 0]
            
            avg_resilience_score = sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0
            avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        for result in results:
            status_emoji = {
                ExperimentStatus.SUCCESS: "✅",
                ExperimentStatus.FAILED: "❌",
                ExperimentStatus.TIMEOUT: "⏰",
                ExperimentStatus.ABORTED: "🛑"
            }.get(result.status, "❓")
            
            table.add_row(
                result.config.name,
                result.config.experiment_type.value,
                f"{status_emoji} {result.status.value}",
                f"{result.duration:.1f}s",
                f"{result.resilience_score:.1f}/100" if result.resilience_score > 0 else "N/A",
                f"{result.recovery_time:.1f}s" if result.recovery_time > 0 else "N/A"
            )
        
        self.console.print("\n")
        self.console.print(table)
        
        # Display overall summary
        summary_panel = Panel.fit(
            f"[bold]Chaos Engineering Summary[/bold]\n"
            f"Total Experiments: {total_experiments}\n"
            f"Successful: {successful_experiments}\n"
            f"Failed: {failed_experiments}\n"
            f"Success Rate: {(successful_experiments/total_experiments)*100:.1f}%\n"
            f"Average Resilience Score: {avg_resilience_score:.1f}/100\n"
            f"Average Recovery Time: {avg_recovery_time:.1f}s",
            title="Overall Results",
            border_style="green" if failed_experiments == 0 else "yellow" if failed_experiments < total_experiments/2 else "red"
        )
        
        self.console.print(summary_panel)
        
        # Display top recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            # Count recommendation frequency
            rec_counts = {}
            for rec in all_recommendations:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            
            # Sort by frequency
            top_recommendations = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            self.console.print("\n[bold blue]🔧 Top Recommendations:[/bold blue]")
            for i, (rec, count) in enumerate(top_recommendations, 1):
                self.console.print(f"  {i}. {rec} (mentioned {count} times)")
        
        # Save detailed report
        await self._save_detailed_report(results)
    
    async def _save_detailed_report(self, results: List[ExperimentResult]):
        """Save detailed chaos engineering report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"chaos_engineering_report_{timestamp}.json"
        
        report_data = {
            "chaos_engineering_report": {
                "timestamp": datetime.now().isoformat(),
                "total_experiments": len(results),
                "successful_experiments": len([r for r in results if r.status == ExperimentStatus.SUCCESS]),
                "failed_experiments": len([r for r in results if r.status == ExperimentStatus.FAILED])
            },
            "experiment_results": [],
            "summary_metrics": {},
            "recommendations": []
        }
        
        # Add experiment results
        for result in results:
            experiment_data = {
                "experiment_id": result.experiment_id,
                "name": result.config.name,
                "type": result.config.experiment_type.value,
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration": result.duration,
                "resilience_score": result.resilience_score,
                "recovery_time": result.recovery_time,
                "metrics": result.metrics,
                "observations": result.observations,
                "recommendations": result.recommendations,
                "error_message": result.error_message if result.error_message else None
            }
            report_data["experiment_results"].append(experiment_data)
        
        # Calculate summary metrics
        if results:
            resilience_scores = [r.resilience_score for r in results if r.resilience_score > 0]
            recovery_times = [r.recovery_time for r in results if r.recovery_time > 0]
            
            report_data["summary_metrics"] = {
                "average_resilience_score": sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0,
                "average_recovery_time": sum(recovery_times) / len(recovery_times) if recovery_times else 0,
                "success_rate": (len([r for r in results if r.status == ExperimentStatus.SUCCESS]) / len(results)) * 100
            }
        
        # Collect all recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Count and sort recommendations
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        report_data["recommendations"] = [
            {"recommendation": rec, "frequency": count}
            for rec, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Save to file
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.console.print(f"\n[dim]Detailed chaos engineering report saved to: {report_file}[/dim]")

# CLI Interface
async def main():
    """Main CLI interface for chaos engineering"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chaos Engineering Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chaos_engineering.py --config experiments.yaml
  python chaos_engineering.py --quick-test
  python chaos_engineering.py --experiment cpu_stress --duration 60 --intensity 0.7
        """
    )
    
    parser.add_argument(
        "--config",
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--experiment",
        choices=[e.value for e in ExperimentType],
        help="Single experiment type to run"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Experiment duration in seconds"
    )
    
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.5,
        help="Experiment intensity (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (use with caution)"
    )
    
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Filter experiments by tags"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test suite"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        orchestrator = ChaosOrchestrator()
        
        if args.quick_test:
            # Add quick test experiments
            quick_experiments = [
                ExperimentConfig(
                    name="quick_cpu_test",
                    experiment_type=ExperimentType.CPU_STRESS,
                    duration=30,
                    intensity=0.3,
                    tags=["quick", "cpu"]
                ),
                ExperimentConfig(
                    name="quick_memory_test",
                    experiment_type=ExperimentType.MEMORY_PRESSURE,
                    duration=20,
                    intensity=0.2,
                    tags=["quick", "memory"]
                )
            ]
            
            for exp in quick_experiments:
                orchestrator.add_experiment(exp)
                
        elif args.config:
            orchestrator.load_config(args.config)
            
        elif args.experiment:
            # Single experiment
            config = ExperimentConfig(
                name=f"single_{args.experiment}",
                experiment_type=ExperimentType(args.experiment),
                duration=args.duration,
                intensity=args.intensity
            )
            orchestrator.add_experiment(config)
        
        else:
            print("Error: Must specify --config, --experiment, or --quick-test")
            return 1
        
        # Run experiments
        results = await orchestrator.run_experiments(
            parallel=args.parallel,
            tags=args.tags
        )
        
        # Determine exit code
        failed_experiments = [r for r in results if r.status == ExperimentStatus.FAILED]
        return 1 if failed_experiments else 0
        
    except KeyboardInterrupt:
        print("\nChaos engineering interrupted by user")
        return 130
    except Exception as e:
        print(f"Chaos engineering failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))