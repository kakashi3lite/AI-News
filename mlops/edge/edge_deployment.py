#!/usr/bin/env python3
"""
Dr. NewsForge's Advanced Edge Computing System

Implements distributed edge computing for real-time news processing,
model inference, and federated learning across edge devices.

Features:
- Edge model deployment and optimization
- Real-time news processing at the edge
- Federated learning coordination
- Model compression and quantization
- Edge-cloud synchronization
- Offline capability with local caching
- Resource-aware computation
- Privacy-preserving processing

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.quantization import quantize_dynamic
from torch.jit import script, trace
import torchvision.transforms as transforms
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    DistilBertModel, DistilBertConfig,
    MobileBertModel, MobileBertConfig
)
import onnx
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import redis
import sqlite3
import requests
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import mlflow
from cryptography.fernet import Fernet
import hashlib
import zlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
EDGE_REQUESTS = Counter('edge_requests_total', 'Total edge requests', ['device_id', 'model_type'])
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency', ['model_type'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy', ['model_type', 'device_id'])
RESOURCE_USAGE = Gauge('resource_usage_percent', 'Resource usage', ['resource_type', 'device_id'])
SYNC_STATUS = Gauge('sync_status', 'Synchronization status', ['device_id'])

@dataclass
class EdgeDevice:
    """Edge device configuration and capabilities."""
    device_id: str
    device_type: str  # mobile, tablet, laptop, edge_server
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    storage_gb: float
    network_bandwidth_mbps: float
    battery_level: Optional[float]  # For mobile devices
    location: Optional[Tuple[float, float]]  # (lat, lon)
    capabilities: List[str]  # ['inference', 'training', 'data_collection']
    last_seen: datetime
    status: str  # 'online', 'offline', 'busy'

@dataclass
class EdgeModel:
    """Edge-optimized model configuration."""
    model_id: str
    model_type: str  # 'summarization', 'classification', 'sentiment'
    framework: str  # 'pytorch', 'onnx', 'tensorrt'
    model_size_mb: float
    inference_time_ms: float
    accuracy: float
    quantized: bool
    compressed: bool
    target_devices: List[str]
    version: str
    checksum: str

@dataclass
class EdgeTask:
    """Task for edge processing."""
    task_id: str
    task_type: str  # 'inference', 'training', 'data_sync'
    priority: int  # 1-10, higher is more urgent
    data: Dict[str, Any]
    target_device: Optional[str]
    deadline: Optional[datetime]
    created_at: datetime
    status: str  # 'pending', 'assigned', 'processing', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None

class ModelOptimizer:
    """Optimizes models for edge deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_cache = {}
        
    def optimize_for_edge(self, 
                         model: nn.Module, 
                         target_device: EdgeDevice,
                         optimization_level: str = 'balanced') -> Dict[str, Any]:
        """Optimize model for specific edge device."""
        
        cache_key = f"{model.__class__.__name__}_{target_device.device_type}_{optimization_level}"
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        optimizations = {
            'original_size': self._get_model_size(model),
            'optimized_models': {},
            'performance_metrics': {}
        }
        
        # Quantization
        if optimization_level in ['aggressive', 'balanced']:
            quantized_model = self._quantize_model(model, target_device)
            optimizations['optimized_models']['quantized'] = quantized_model
            optimizations['performance_metrics']['quantized'] = self._benchmark_model(
                quantized_model, target_device
            )
        
        # Pruning
        if optimization_level == 'aggressive':
            pruned_model = self._prune_model(model, target_device)
            optimizations['optimized_models']['pruned'] = pruned_model
            optimizations['performance_metrics']['pruned'] = self._benchmark_model(
                pruned_model, target_device
            )
        
        # Knowledge distillation
        if target_device.memory_gb < 4 and optimization_level in ['aggressive', 'balanced']:
            distilled_model = self._distill_model(model, target_device)
            optimizations['optimized_models']['distilled'] = distilled_model
            optimizations['performance_metrics']['distilled'] = self._benchmark_model(
                distilled_model, target_device
            )
        
        # ONNX conversion
        if target_device.device_type in ['mobile', 'tablet']:
            onnx_model = self._convert_to_onnx(model)
            optimizations['optimized_models']['onnx'] = onnx_model
            optimizations['performance_metrics']['onnx'] = self._benchmark_onnx_model(
                onnx_model, target_device
            )
        
        # TensorRT optimization (for devices with compatible GPUs)
        if target_device.gpu_available and target_device.gpu_memory_gb > 2:
            try:
                tensorrt_model = self._convert_to_tensorrt(model)
                optimizations['optimized_models']['tensorrt'] = tensorrt_model
                optimizations['performance_metrics']['tensorrt'] = self._benchmark_tensorrt_model(
                    tensorrt_model, target_device
                )
            except Exception as e:
                logger.warning(f"TensorRT optimization failed: {e}")
        
        # Select best optimization
        best_model = self._select_best_optimization(optimizations, target_device)
        optimizations['recommended'] = best_model
        
        self.optimization_cache[cache_key] = optimizations
        return optimizations
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 * 1024)
    
    def _quantize_model(self, model: nn.Module, target_device: EdgeDevice) -> nn.Module:
        """Apply dynamic quantization to model."""
        model.eval()
        
        # Dynamic quantization for CPU inference
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _prune_model(self, model: nn.Module, target_device: EdgeDevice) -> nn.Module:
        """Apply structured pruning to model."""
        import torch.nn.utils.prune as prune
        
        # Clone model for pruning
        pruned_model = type(model)()
        pruned_model.load_state_dict(model.state_dict())
        
        # Apply magnitude-based pruning
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=0.3)
                prune.remove(module, 'weight')
        
        return pruned_model
    
    def _distill_model(self, teacher_model: nn.Module, target_device: EdgeDevice) -> nn.Module:
        """Create distilled version of model."""
        # Simplified distillation - create smaller student model
        # In practice, this would involve training a student model
        
        if hasattr(teacher_model, 'config'):
            # For transformer models, create smaller version
            config = teacher_model.config
            student_config = type(config)(
                hidden_size=config.hidden_size // 2,
                num_hidden_layers=config.num_hidden_layers // 2,
                num_attention_heads=config.num_attention_heads // 2,
                intermediate_size=config.intermediate_size // 2
            )
            student_model = type(teacher_model)(student_config)
        else:
            # For other models, return a simplified version
            student_model = teacher_model
        
        return student_model
    
    def _convert_to_onnx(self, model: nn.Module) -> str:
        """Convert PyTorch model to ONNX format."""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 512)  # Adjust based on model input
        
        # Export to ONNX
        onnx_path = f"/tmp/model_{int(time.time())}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return onnx_path
    
    def _convert_to_tensorrt(self, model: nn.Module) -> str:
        """Convert model to TensorRT format."""
        # First convert to ONNX
        onnx_path = self._convert_to_onnx(model)
        
        # Convert ONNX to TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                raise RuntimeError("Failed to parse ONNX model")
        
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        # Save engine
        trt_path = f"/tmp/model_{int(time.time())}.trt"
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())
        
        return trt_path
    
    def _benchmark_model(self, model: nn.Module, target_device: EdgeDevice) -> Dict[str, float]:
        """Benchmark model performance."""
        model.eval()
        
        # Create test input
        test_input = torch.randn(1, 512)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = model(test_input)
            times.append(time.time() - start_time)
        
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'p95_inference_time_ms': np.percentile(times, 95) * 1000,
            'model_size_mb': self._get_model_size(model),
            'memory_usage_mb': self._estimate_memory_usage(model)
        }
    
    def _benchmark_onnx_model(self, onnx_path: str, target_device: EdgeDevice) -> Dict[str, float]:
        """Benchmark ONNX model performance."""
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Create test input
        input_name = session.get_inputs()[0].name
        test_input = {input_name: np.random.randn(1, 512).astype(np.float32)}
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, test_input)
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            _ = session.run(None, test_input)
            times.append(time.time() - start_time)
        
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'p95_inference_time_ms': np.percentile(times, 95) * 1000,
            'model_size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
            'memory_usage_mb': 0  # ONNX Runtime manages memory
        }
    
    def _benchmark_tensorrt_model(self, trt_path: str, target_device: EdgeDevice) -> Dict[str, float]:
        """Benchmark TensorRT model performance."""
        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(trt_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Allocate GPU memory
        input_shape = (1, 512)
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = input_size  # Assume same size for simplicity
        
        h_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)
        h_output = cuda.pagelocked_empty(input_shape, dtype=np.float32)
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        stream = cuda.Stream()
        
        # Warmup
        for _ in range(10):
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            times.append(time.time() - start_time)
        
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'p95_inference_time_ms': np.percentile(times, 95) * 1000,
            'model_size_mb': os.path.getsize(trt_path) / (1024 * 1024),
            'memory_usage_mb': (input_size + output_size) / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate memory usage in MB."""
        # Simplified estimation
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Add estimated activation memory (rough approximation)
        activation_memory = param_memory * 2  # Rough estimate
        
        total_memory = param_memory + buffer_memory + activation_memory
        return total_memory / (1024 * 1024)
    
    def _select_best_optimization(self, 
                                optimizations: Dict[str, Any], 
                                target_device: EdgeDevice) -> str:
        """Select best optimization based on device constraints."""
        
        performance_metrics = optimizations['performance_metrics']
        
        # Define scoring function based on device constraints
        def score_optimization(opt_name: str, metrics: Dict[str, float]) -> float:
            score = 0.0
            
            # Latency score (lower is better)
            latency_ms = metrics.get('avg_inference_time_ms', 1000)
            latency_score = max(0, 1 - latency_ms / 1000)  # Normalize to [0, 1]
            score += latency_score * 0.4
            
            # Memory score (lower is better)
            memory_mb = metrics.get('memory_usage_mb', 1000)
            memory_limit = target_device.memory_gb * 1024 * 0.5  # Use 50% of available memory
            memory_score = max(0, 1 - memory_mb / memory_limit)
            score += memory_score * 0.3
            
            # Model size score (lower is better)
            size_mb = metrics.get('model_size_mb', 1000)
            size_limit = target_device.storage_gb * 1024 * 0.1  # Use 10% of storage
            size_score = max(0, 1 - size_mb / size_limit)
            score += size_score * 0.3
            
            return score
        
        best_optimization = 'original'
        best_score = 0.0
        
        for opt_name, metrics in performance_metrics.items():
            score = score_optimization(opt_name, metrics)
            if score > best_score:
                best_score = score
                best_optimization = opt_name
        
        return best_optimization

class EdgeOrchestrator:
    """Orchestrates edge computing tasks and model deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.devices = {}  # device_id -> EdgeDevice
        self.models = {}   # model_id -> EdgeModel
        self.tasks = deque(maxlen=10000)  # Task queue
        self.task_assignments = {}  # task_id -> device_id
        
        # Components
        self.optimizer = ModelOptimizer(config)
        
        # Redis for coordination
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Local database for offline capability
        self.db_path = config.get('db_path', '/tmp/edge_orchestrator.db')
        self._init_database()
        
        # Encryption for secure communication
        self.encryption_key = config.get('encryption_key', Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)
        
        # Task scheduling
        self.scheduler_thread = threading.Thread(target=self._task_scheduler, daemon=True)
        self.scheduler_running = True
        self.scheduler_thread.start()
        
        # Device monitoring
        self.monitor_thread = threading.Thread(target=self._device_monitor, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Edge orchestrator initialized")
    
    def _init_database(self):
        """Initialize local SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                device_data TEXT,
                last_updated TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_data TEXT,
                model_file BLOB,
                last_updated TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_data TEXT,
                status TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_device(self, device: EdgeDevice) -> bool:
        """Register a new edge device."""
        try:
            self.devices[device.device_id] = device
            
            # Store in Redis
            device_data = json.dumps(asdict(device), default=str)
            self.redis_client.hset('edge_devices', device.device_id, device_data)
            
            # Store in local database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO devices (device_id, device_data, last_updated) VALUES (?, ?, ?)',
                (device.device_id, device_data, datetime.now())
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Registered device {device.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register device {device.device_id}: {e}")
            return False
    
    def deploy_model(self, model_id: str, target_devices: List[str]) -> Dict[str, bool]:
        """Deploy model to target edge devices."""
        deployment_results = {}
        
        for device_id in target_devices:
            if device_id not in self.devices:
                deployment_results[device_id] = False
                continue
            
            device = self.devices[device_id]
            
            try:
                # Optimize model for device
                # In practice, load the actual model here
                dummy_model = nn.Linear(512, 256)  # Placeholder
                optimizations = self.optimizer.optimize_for_edge(dummy_model, device)
                
                # Create deployment task
                task = EdgeTask(
                    task_id=f"deploy_{model_id}_{device_id}_{int(time.time())}",
                    task_type='model_deployment',
                    priority=5,
                    data={
                        'model_id': model_id,
                        'optimizations': optimizations,
                        'target_device': device_id
                    },
                    target_device=device_id,
                    deadline=datetime.now() + timedelta(minutes=30),
                    created_at=datetime.now(),
                    status='pending'
                )
                
                self.submit_task(task)
                deployment_results[device_id] = True
                
            except Exception as e:
                logger.error(f"Failed to deploy model {model_id} to device {device_id}: {e}")
                deployment_results[device_id] = False
        
        return deployment_results
    
    def submit_task(self, task: EdgeTask) -> bool:
        """Submit task for edge processing."""
        try:
            self.tasks.append(task)
            
            # Store in Redis
            task_data = json.dumps(asdict(task), default=str)
            self.redis_client.lpush('edge_tasks', task_data)
            
            # Store in local database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO tasks (task_id, task_data, status, created_at) VALUES (?, ?, ?, ?)',
                (task.task_id, task_data, task.status, task.created_at)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Submitted task {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def _task_scheduler(self):
        """Background task scheduler."""
        while self.scheduler_running:
            try:
                # Get pending tasks
                pending_tasks = [t for t in self.tasks if t.status == 'pending']
                
                # Sort by priority and deadline
                pending_tasks.sort(key=lambda t: (-t.priority, t.deadline or datetime.max))
                
                for task in pending_tasks:
                    # Find suitable device
                    suitable_device = self._find_suitable_device(task)
                    
                    if suitable_device:
                        # Assign task
                        task.status = 'assigned'
                        self.task_assignments[task.task_id] = suitable_device.device_id
                        
                        # Send task to device
                        self._send_task_to_device(task, suitable_device)
                        
                        logger.info(f"Assigned task {task.task_id} to device {suitable_device.device_id}")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in task scheduler: {e}")
                time.sleep(5)
    
    def _find_suitable_device(self, task: EdgeTask) -> Optional[EdgeDevice]:
        """Find suitable device for task execution."""
        # Filter available devices
        available_devices = [
            device for device in self.devices.values()
            if device.status == 'online' and self._can_handle_task(device, task)
        ]
        
        if not available_devices:
            return None
        
        # Score devices based on suitability
        def score_device(device: EdgeDevice) -> float:
            score = 0.0
            
            # Resource availability
            cpu_usage = self._get_device_cpu_usage(device.device_id)
            memory_usage = self._get_device_memory_usage(device.device_id)
            
            score += (1 - cpu_usage) * 0.3
            score += (1 - memory_usage) * 0.3
            
            # Device capabilities
            if task.task_type in device.capabilities:
                score += 0.2
            
            # Network proximity (simplified)
            if device.location and task.data.get('location'):
                # Calculate distance and adjust score
                distance = self._calculate_distance(device.location, task.data['location'])
                score += max(0, 1 - distance / 1000) * 0.2  # Normalize by 1000km
            
            return score
        
        # Select best device
        best_device = max(available_devices, key=score_device)
        return best_device
    
    def _can_handle_task(self, device: EdgeDevice, task: EdgeTask) -> bool:
        """Check if device can handle the task."""
        # Check capabilities
        if task.task_type == 'inference' and 'inference' not in device.capabilities:
            return False
        
        if task.task_type == 'training' and 'training' not in device.capabilities:
            return False
        
        # Check resource requirements
        if task.data.get('min_memory_gb', 0) > device.memory_gb:
            return False
        
        if task.data.get('requires_gpu', False) and not device.gpu_available:
            return False
        
        # Check current load
        current_tasks = sum(1 for t in self.tasks if 
                          self.task_assignments.get(t.task_id) == device.device_id and 
                          t.status in ['assigned', 'processing'])
        
        max_concurrent = device.cpu_cores
        if current_tasks >= max_concurrent:
            return False
        
        return True
    
    def _send_task_to_device(self, task: EdgeTask, device: EdgeDevice):
        """Send task to edge device for execution."""
        try:
            # Encrypt task data
            task_json = json.dumps(asdict(task), default=str)
            encrypted_data = self.cipher.encrypt(task_json.encode())
            
            # Send via Redis pub/sub
            channel = f"device_{device.device_id}_tasks"
            self.redis_client.publish(channel, encrypted_data.decode())
            
            # Update metrics
            EDGE_REQUESTS.labels(
                device_id=device.device_id,
                model_type=task.data.get('model_type', 'unknown')
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to send task {task.task_id} to device {device.device_id}: {e}")
            task.status = 'failed'
    
    def _device_monitor(self):
        """Monitor device health and status."""
        while self.scheduler_running:
            try:
                for device_id, device in self.devices.items():
                    # Check device heartbeat
                    last_heartbeat = self.redis_client.hget('device_heartbeats', device_id)
                    
                    if last_heartbeat:
                        last_seen = datetime.fromisoformat(last_heartbeat)
                        if (datetime.now() - last_seen).total_seconds() > 60:  # 1 minute timeout
                            device.status = 'offline'
                        else:
                            device.status = 'online'
                    
                    # Update resource usage metrics
                    cpu_usage = self._get_device_cpu_usage(device_id)
                    memory_usage = self._get_device_memory_usage(device_id)
                    
                    RESOURCE_USAGE.labels(
                        resource_type='cpu',
                        device_id=device_id
                    ).set(cpu_usage)
                    
                    RESOURCE_USAGE.labels(
                        resource_type='memory',
                        device_id=device_id
                    ).set(memory_usage)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in device monitor: {e}")
                time.sleep(60)
    
    def _get_device_cpu_usage(self, device_id: str) -> float:
        """Get device CPU usage from Redis or estimate."""
        usage_str = self.redis_client.hget('device_cpu_usage', device_id)
        if usage_str:
            return float(usage_str)
        return 0.5  # Default estimate
    
    def _get_device_memory_usage(self, device_id: str) -> float:
        """Get device memory usage from Redis or estimate."""
        usage_str = self.redis_client.hget('device_memory_usage', device_id)
        if usage_str:
            return float(usage_str)
        return 0.5  # Default estimate
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two locations in km."""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = radians(loc1[0]), radians(loc1[1])
        lat2, lon2 = radians(loc2[0]), radians(loc2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return 6371 * c  # Earth radius in km
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        online_devices = sum(1 for d in self.devices.values() if d.status == 'online')
        total_devices = len(self.devices)
        
        pending_tasks = sum(1 for t in self.tasks if t.status == 'pending')
        processing_tasks = sum(1 for t in self.tasks if t.status == 'processing')
        completed_tasks = sum(1 for t in self.tasks if t.status == 'completed')
        
        return {
            'devices': {
                'total': total_devices,
                'online': online_devices,
                'offline': total_devices - online_devices
            },
            'tasks': {
                'pending': pending_tasks,
                'processing': processing_tasks,
                'completed': completed_tasks
            },
            'models': {
                'deployed': len(self.models)
            },
            'system_health': 'healthy' if online_devices > 0 else 'degraded'
        }

class EdgeAgent:
    """Edge agent running on individual devices."""
    
    def __init__(self, device: EdgeDevice, config: Dict[str, Any]):
        self.device = device
        self.config = config
        
        # Local model storage
        self.models = {}
        self.model_cache = {}
        
        # Task processing
        self.current_tasks = {}
        self.task_history = deque(maxlen=1000)
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Encryption
        self.encryption_key = config.get('encryption_key')
        self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
        
        # Local storage
        self.storage_path = Path(config.get('storage_path', f'/tmp/edge_agent_{device.device_id}'))
        self.storage_path.mkdir(exist_ok=True)
        
        # Performance monitoring
        self.performance_metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_processing_time': 0.0,
            'last_update': datetime.now()
        }
        
        # Start background processes
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.task_listener_thread = threading.Thread(target=self._task_listener, daemon=True)
        self.resource_monitor_thread = threading.Thread(target=self._resource_monitor, daemon=True)
        
        self.heartbeat_thread.start()
        self.task_listener_thread.start()
        self.resource_monitor_thread.start()
        
        logger.info(f"Edge agent started for device {device.device_id}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat to orchestrator."""
        while self.running:
            try:
                # Send heartbeat
                self.redis_client.hset(
                    'device_heartbeats',
                    self.device.device_id,
                    datetime.now().isoformat()
                )
                
                # Update device status
                device_data = json.dumps(asdict(self.device), default=str)
                self.redis_client.hset('edge_devices', self.device.device_id, device_data)
                
                time.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(60)
    
    def _task_listener(self):
        """Listen for incoming tasks."""
        channel = f"device_{self.device.device_id}_tasks"
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(channel)
        
        for message in pubsub.listen():
            if not self.running:
                break
                
            if message['type'] == 'message':
                try:
                    # Decrypt task data
                    if self.cipher:
                        encrypted_data = message['data'].encode()
                        decrypted_data = self.cipher.decrypt(encrypted_data)
                        task_data = json.loads(decrypted_data.decode())
                    else:
                        task_data = json.loads(message['data'])
                    
                    # Create task object
                    task = EdgeTask(**task_data)
                    
                    # Process task
                    self._process_task(task)
                    
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
    
    def _process_task(self, task: EdgeTask):
        """Process an edge task."""
        start_time = time.time()
        task.status = 'processing'
        self.current_tasks[task.task_id] = task
        
        try:
            if task.task_type == 'inference':
                result = self._run_inference(task)
            elif task.task_type == 'model_deployment':
                result = self._deploy_model(task)
            elif task.task_type == 'data_sync':
                result = self._sync_data(task)
            elif task.task_type == 'training':
                result = self._run_training(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.result = result
            task.status = 'completed'
            self.performance_metrics['successful_tasks'] += 1
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.result = {'error': str(e)}
            task.status = 'failed'
            self.performance_metrics['failed_tasks'] += 1
        
        finally:
            processing_time = time.time() - start_time
            
            # Update metrics
            self.performance_metrics['total_tasks'] += 1
            self.performance_metrics['avg_processing_time'] = (
                (self.performance_metrics['avg_processing_time'] * 
                 (self.performance_metrics['total_tasks'] - 1) + processing_time) /
                self.performance_metrics['total_tasks']
            )
            
            # Record metrics
            INFERENCE_LATENCY.labels(
                model_type=task.data.get('model_type', 'unknown')
            ).observe(processing_time)
            
            # Move to history
            self.task_history.append(task)
            del self.current_tasks[task.task_id]
            
            # Send result back
            self._send_task_result(task)
    
    def _run_inference(self, task: EdgeTask) -> Dict[str, Any]:
        """Run model inference."""
        model_id = task.data.get('model_id')
        input_data = task.data.get('input_data')
        
        if not model_id or not input_data:
            raise ValueError("Missing model_id or input_data")
        
        # Load model if not cached
        if model_id not in self.model_cache:
            self._load_model(model_id)
        
        model = self.model_cache[model_id]
        
        # Prepare input
        if isinstance(input_data, str):
            # Text input - tokenize
            inputs = self._tokenize_text(input_data)
        else:
            inputs = input_data
        
        # Run inference
        with torch.no_grad():
            outputs = model(inputs)
        
        # Process outputs
        if hasattr(outputs, 'logits'):
            predictions = torch.softmax(outputs.logits, dim=-1)
            result = predictions.cpu().numpy().tolist()
        else:
            result = outputs.cpu().numpy().tolist()
        
        return {
            'predictions': result,
            'model_id': model_id,
            'processing_time_ms': task.data.get('processing_time', 0)
        }
    
    def _deploy_model(self, task: EdgeTask) -> Dict[str, Any]:
        """Deploy model to device."""
        model_id = task.data.get('model_id')
        optimizations = task.data.get('optimizations', {})
        
        # Get recommended optimization
        recommended = optimizations.get('recommended', 'original')
        
        if recommended in optimizations.get('optimized_models', {}):
            model_path = optimizations['optimized_models'][recommended]
            
            # Load and cache model
            if recommended == 'onnx':
                model = ort.InferenceSession(model_path)
            elif recommended == 'tensorrt':
                # Load TensorRT model
                model = self._load_tensorrt_model(model_path)
            else:
                # Load PyTorch model
                model = torch.jit.load(model_path)
            
            self.model_cache[model_id] = model
            self.models[model_id] = {
                'model_id': model_id,
                'optimization': recommended,
                'deployed_at': datetime.now(),
                'performance_metrics': optimizations.get('performance_metrics', {}).get(recommended, {})
            }
            
            return {
                'status': 'deployed',
                'model_id': model_id,
                'optimization': recommended,
                'performance_metrics': self.models[model_id]['performance_metrics']
            }
        
        raise ValueError(f"No suitable optimization found for model {model_id}")
    
    def _sync_data(self, task: EdgeTask) -> Dict[str, Any]:
        """Synchronize data with cloud."""
        sync_type = task.data.get('sync_type', 'bidirectional')
        data_types = task.data.get('data_types', ['models', 'metrics'])
        
        synced_items = []
        
        for data_type in data_types:
            if data_type == 'models':
                # Sync model updates
                synced_items.extend(self._sync_models())
            elif data_type == 'metrics':
                # Sync performance metrics
                synced_items.extend(self._sync_metrics())
            elif data_type == 'tasks':
                # Sync task history
                synced_items.extend(self._sync_task_history())
        
        return {
            'synced_items': synced_items,
            'sync_type': sync_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_training(self, task: EdgeTask) -> Dict[str, Any]:
        """Run federated learning training."""
        model_id = task.data.get('model_id')
        training_data = task.data.get('training_data')
        global_model_weights = task.data.get('global_weights')
        
        if not model_id or not training_data:
            raise ValueError("Missing model_id or training_data")
        
        # Load model
        if model_id not in self.model_cache:
            self._load_model(model_id)
        
        model = self.model_cache[model_id]
        
        # Update with global weights if provided
        if global_model_weights:
            model.load_state_dict(global_model_weights)
        
        # Prepare training data
        train_loader = self._prepare_training_data(training_data)
        
        # Train model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch['input'])
            loss = F.cross_entropy(outputs, batch['target'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Get updated weights
        updated_weights = model.state_dict()
        
        return {
            'updated_weights': updated_weights,
            'training_loss': avg_loss,
            'num_samples': len(training_data),
            'model_id': model_id
        }
    
    def _load_model(self, model_id: str):
        """Load model from storage."""
        model_path = self.storage_path / f"{model_id}.pt"
        
        if model_path.exists():
            model = torch.jit.load(str(model_path))
            self.model_cache[model_id] = model
        else:
            # Download from cloud/orchestrator
            self._download_model(model_id)
    
    def _download_model(self, model_id: str):
        """Download model from orchestrator."""
        # Simplified download - in practice, this would fetch from cloud storage
        logger.info(f"Downloading model {model_id}")
        
        # Create dummy model for demonstration
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Save to local storage
        model_path = self.storage_path / f"{model_id}.pt"
        torch.jit.save(torch.jit.script(model), str(model_path))
        
        self.model_cache[model_id] = model
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text input."""
        # Simplified tokenization
        # In practice, use proper tokenizer
        words = text.split()
        token_ids = [hash(word) % 10000 for word in words]
        
        # Pad or truncate to fixed length
        max_length = 512
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([0] * (max_length - len(token_ids)))
        
        return torch.tensor([token_ids], dtype=torch.long)
    
    def _load_tensorrt_model(self, model_path: str):
        """Load TensorRT model."""
        # Simplified TensorRT loading
        logger.info(f"Loading TensorRT model from {model_path}")
        return None  # Placeholder
    
    def _sync_models(self) -> List[str]:
        """Sync model updates."""
        synced = []
        for model_id in self.models:
            # Upload model metrics
            metrics_key = f"model_metrics_{self.device.device_id}_{model_id}"
            metrics_data = json.dumps(self.models[model_id])
            self.redis_client.set(metrics_key, metrics_data)
            synced.append(f"model_metrics_{model_id}")
        
        return synced
    
    def _sync_metrics(self) -> List[str]:
        """Sync performance metrics."""
        metrics_key = f"device_metrics_{self.device.device_id}"
        metrics_data = json.dumps(self.performance_metrics, default=str)
        self.redis_client.set(metrics_key, metrics_data)
        
        return [f"device_metrics_{self.device.device_id}"]
    
    def _sync_task_history(self) -> List[str]:
        """Sync task execution history."""
        history_key = f"task_history_{self.device.device_id}"
        
        # Convert task history to serializable format
        history_data = []
        for task in self.task_history:
            history_data.append({
                'task_id': task.task_id,
                'task_type': task.task_type,
                'status': task.status,
                'created_at': task.created_at.isoformat(),
                'processing_time': getattr(task, 'processing_time', 0)
            })
        
        self.redis_client.set(history_key, json.dumps(history_data))
        return [f"task_history_{self.device.device_id}"]
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> DataLoader:
        """Prepare training data for federated learning."""
        # Simplified data preparation
        inputs = []
        targets = []
        
        for sample in training_data:
            inputs.append(sample.get('input', [0] * 512))
            targets.append(sample.get('target', 0))
        
        # Create dataset
        dataset = [{
            'input': torch.tensor(inp, dtype=torch.float32),
            'target': torch.tensor(tgt, dtype=torch.long)
        } for inp, tgt in zip(inputs, targets)]
        
        return DataLoader(dataset, batch_size=32, shuffle=True)
    
    def _send_task_result(self, task: EdgeTask):
        """Send task result back to orchestrator."""
        try:
            result_data = {
                'task_id': task.task_id,
                'device_id': self.device.device_id,
                'status': task.status,
                'result': task.result,
                'completed_at': datetime.now().isoformat()
            }
            
            # Encrypt result if cipher available
            if self.cipher:
                result_json = json.dumps(result_data)
                encrypted_result = self.cipher.encrypt(result_json.encode())
                self.redis_client.lpush('task_results', encrypted_result.decode())
            else:
                self.redis_client.lpush('task_results', json.dumps(result_data))
            
        except Exception as e:
            logger.error(f"Failed to send task result: {e}")
    
    def _resource_monitor(self):
        """Monitor device resources."""
        while self.running:
            try:
                # Get CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent / 100.0
                
                # Get GPU usage if available
                gpu_usage = 0.0
                if self.device.gpu_available:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_usage = gpus[0].load
                    except:
                        pass
                
                # Update Redis
                self.redis_client.hset(
                    'device_cpu_usage',
                    self.device.device_id,
                    cpu_usage / 100.0
                )
                
                self.redis_client.hset(
                    'device_memory_usage',
                    self.device.device_id,
                    memory_usage
                )
                
                if gpu_usage > 0:
                    self.redis_client.hset(
                        'device_gpu_usage',
                        self.device.device_id,
                        gpu_usage
                    )
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(60)
    
    def stop(self):
        """Stop the edge agent."""
        self.running = False
        logger.info(f"Edge agent stopped for device {self.device.device_id}")

def create_edge_api(orchestrator: EdgeOrchestrator) -> Flask:
    """Create Flask API for edge orchestrator."""
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @app.route('/api/devices', methods=['GET'])
    def get_devices():
        """Get all registered devices."""
        devices = [asdict(device) for device in orchestrator.devices.values()]
        return jsonify(devices)
    
    @app.route('/api/devices', methods=['POST'])
    def register_device():
        """Register a new device."""
        data = request.json
        device = EdgeDevice(**data)
        success = orchestrator.register_device(device)
        
        return jsonify({'success': success}), 200 if success else 400
    
    @app.route('/api/models/<model_id>/deploy', methods=['POST'])
    def deploy_model(model_id):
        """Deploy model to devices."""
        data = request.json
        target_devices = data.get('target_devices', [])
        
        results = orchestrator.deploy_model(model_id, target_devices)
        return jsonify(results)
    
    @app.route('/api/tasks', methods=['POST'])
    def submit_task():
        """Submit a new task."""
        data = request.json
        task = EdgeTask(**data)
        success = orchestrator.submit_task(task)
        
        return jsonify({'success': success, 'task_id': task.task_id}), 200 if success else 400
    
    @app.route('/api/status', methods=['GET'])
    def get_status():
        """Get system status."""
        status = orchestrator.get_system_status()
        return jsonify(status)
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        emit('status', orchestrator.get_system_status())
    
    @socketio.on('request_status')
    def handle_status_request():
        """Handle status request."""
        emit('status', orchestrator.get_system_status())
    
    return app

def main():
    """Main function to run edge computing system."""
    config = {
        'redis_host': os.getenv('REDIS_HOST', 'localhost'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        'encryption_key': os.getenv('EDGE_ENCRYPTION_KEY', Fernet.generate_key())
    }
    
    # Start Prometheus metrics server
    start_http_server(8001)
    
    # Initialize orchestrator
    orchestrator = EdgeOrchestrator(config)
    
    # Create sample devices
    devices = [
        EdgeDevice(
            device_id="mobile_001",
            device_type="mobile",
            cpu_cores=8,
            memory_gb=6,
            gpu_available=True,
            gpu_memory_gb=2,
            storage_gb=128,
            network_bandwidth_mbps=100,
            battery_level=85.0,
            location=(37.7749, -122.4194),  # San Francisco
            capabilities=['inference', 'data_collection'],
            last_seen=datetime.now(),
            status='online'
        ),
        EdgeDevice(
            device_id="edge_server_001",
            device_type="edge_server",
            cpu_cores=16,
            memory_gb=32,
            gpu_available=True,
            gpu_memory_gb=16,
            storage_gb=1000,
            network_bandwidth_mbps=1000,
            battery_level=None,
            location=(40.7128, -74.0060),  # New York
            capabilities=['inference', 'training', 'data_collection'],
            last_seen=datetime.now(),
            status='online'
        )
    ]
    
    # Register devices
    for device in devices:
        orchestrator.register_device(device)
    
    # Create and start API
    app = create_edge_api(orchestrator)
    
    logger.info("Edge computing system started")
    logger.info("API available at http://localhost:5000")
    logger.info("Metrics available at http://localhost:8001")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()