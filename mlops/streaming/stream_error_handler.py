#!/usr/bin/env python3
"""
Advanced Error Handler for Real-Time News Streaming Pipeline

Provides comprehensive error handling, retry mechanisms, circuit breakers,
and fallback strategies specifically designed for streaming data processing.

Features:
- Stream-specific error classification
- Adaptive retry strategies for different error types
- Circuit breaker patterns for external services
- Dead letter queue management
- Performance monitoring and alerting
- Graceful degradation strategies

Author: Dr. Nova "NewsForge" Arclight
Version: 1.0.0
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import traceback
import threading
from functools import wraps

import numpy as np
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
STREAM_ERRORS = Counter('stream_errors_total', 'Total streaming errors', ['error_type', 'component'])
RETRY_ATTEMPTS = Counter('retry_attempts_total', 'Total retry attempts', ['operation', 'attempt'])
CIRCUIIT_BREAKER_STATE = Gauge('circuit_breaker_state', 'Circuit breaker state', ['service'])
FALLBACK_USAGE = Counter('fallback_usage_total', 'Fallback mechanism usage', ['fallback_type'])
ERROR_RECOVERY_TIME = Histogram('error_recovery_time_seconds', 'Time to recover from errors')

class StreamErrorType(Enum):
    """Classification of streaming-specific errors."""
    KAFKA_CONNECTION = "kafka_connection"
    KAFKA_TIMEOUT = "kafka_timeout"
    KAFKA_SERIALIZATION = "kafka_serialization"
    REDIS_CONNECTION = "redis_connection"
    ELASTICSEARCH_CONNECTION = "elasticsearch_connection"
    MODEL_INFERENCE = "model_inference"
    MODEL_LOADING = "model_loading"
    DATA_VALIDATION = "data_validation"
    DATA_CORRUPTION = "data_corruption"
    RATE_LIMIT = "rate_limit"
    MEMORY_OVERFLOW = "memory_overflow"
    PROCESSING_TIMEOUT = "processing_timeout"
    EXTERNAL_API = "external_api"
    NETWORK_TIMEOUT = "network_timeout"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: StreamErrorType
    component: str
    operation: str
    timestamp: datetime
    error_message: str
    stack_trace: str
    metadata: Dict[str, Any]
    retry_count: int = 0
    recovery_time: Optional[float] = None

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: List[StreamErrorType] = None

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    monitor_window: float = 300.0  # 5 minutes

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker implementation for external services."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.failure_times = deque(maxlen=100)
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to CLOSED")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_times.append(self.last_failure_time)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' moved to OPEN")
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' moved to OPEN")
            
            # Update metrics
            CIRCU IT_BREAKER_STATE.labels(service=self.name).set(
                1 if self.state == CircuitBreakerState.OPEN else 0
            )

class StreamErrorHandler:
    """Advanced error handler for streaming pipeline."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history = defaultdict(list)
        self.dead_letter_queue = deque(maxlen=10000)
        self.fallback_strategies: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        logger.info("Stream error handler initialized")
    
    def classify_error(self, error: Exception, component: str = "unknown") -> StreamErrorType:
        """Classify error into appropriate type."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Kafka errors
        if 'kafka' in error_str or 'kafka' in component.lower():
            if 'timeout' in error_str:
                return StreamErrorType.KAFKA_TIMEOUT
            elif 'connection' in error_str or 'connect' in error_str:
                return StreamErrorType.KAFKA_CONNECTION
            elif 'serializ' in error_str:
                return StreamErrorType.KAFKA_SERIALIZATION
        
        # Redis errors
        if 'redis' in error_str or 'redis' in component.lower():
            return StreamErrorType.REDIS_CONNECTION
        
        # Elasticsearch errors
        if 'elasticsearch' in error_str or 'elastic' in component.lower():
            return StreamErrorType.ELASTICSEARCH_CONNECTION
        
        # Model errors
        if 'model' in component.lower() or 'inference' in error_str:
            if 'load' in error_str:
                return StreamErrorType.MODEL_LOADING
            else:
                return StreamErrorType.MODEL_INFERENCE
        
        # Network and API errors
        if 'timeout' in error_str:
            return StreamErrorType.NETWORK_TIMEOUT
        elif 'rate limit' in error_str or 'too many requests' in error_str:
            return StreamErrorType.RATE_LIMIT
        elif 'permission' in error_str or 'unauthorized' in error_str:
            return StreamErrorType.PERMISSION_DENIED
        elif 'memory' in error_str:
            return StreamErrorType.MEMORY_OVERFLOW
        
        # Data errors
        if 'validation' in error_str or 'invalid' in error_str:
            return StreamErrorType.DATA_VALIDATION
        elif 'corrupt' in error_str:
            return StreamErrorType.DATA_CORRUPTION
        
        return StreamErrorType.UNKNOWN
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register a circuit breaker for a service."""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
        logger.info(f"Circuit breaker registered for service: {name}")
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback strategy for an operation."""
        self.fallback_strategies[operation] = fallback_func
        logger.info(f"Fallback strategy registered for operation: {operation}")
    
    async def execute_with_retry(self, 
                                func: Callable,
                                operation: str,
                                component: str = "unknown",
                                retry_config: Optional[RetryConfig] = None,
                                circuit_breaker_name: Optional[str] = None,
                                fallback_operation: Optional[str] = None,
                                **kwargs) -> Any:
        """Execute function with comprehensive error handling."""
        
        if retry_config is None:
            retry_config = RetryConfig()
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                # Use circuit breaker if specified
                if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                    cb = self.circuit_breakers[circuit_breaker_name]
                    if asyncio.iscoroutinefunction(func):
                        result = await cb.call(lambda: asyncio.create_task(func(**kwargs)))
                    else:
                        result = cb.call(func, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**kwargs)
                    else:
                        result = func(**kwargs)
                
                # Record successful recovery if this was a retry
                if attempt > 0:
                    recovery_time = time.time() - start_time
                    ERROR_RECOVERY_TIME.observe(recovery_time)
                    logger.info(f"Operation '{operation}' recovered after {attempt} attempts")
                
                return result
                
            except Exception as error:
                last_error = error
                error_type = self.classify_error(error, component)
                
                # Record error context
                error_context = ErrorContext(
                    error_type=error_type,
                    component=component,
                    operation=operation,
                    timestamp=datetime.now(),
                    error_message=str(error),
                    stack_trace=traceback.format_exc(),
                    metadata=kwargs,
                    retry_count=attempt
                )
                
                self._record_error(error_context)
                
                # Check if error is retryable
                if (retry_config.retryable_errors and 
                    error_type not in retry_config.retryable_errors):
                    logger.error(f"Non-retryable error in '{operation}': {error}")
                    break
                
                # Don't retry on last attempt
                if attempt >= retry_config.max_retries:
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    retry_config.base_delay * (retry_config.exponential_base ** attempt),
                    retry_config.max_delay
                )
                
                if retry_config.jitter:
                    delay *= (0.5 + np.random.random() * 0.5)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{retry_config.max_retries + 1} failed for '{operation}': {error}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                RETRY_ATTEMPTS.labels(operation=operation, attempt=str(attempt + 1)).inc()
                await asyncio.sleep(delay)
        
        # All retries failed, try fallback
        if fallback_operation and fallback_operation in self.fallback_strategies:
            try:
                logger.info(f"Executing fallback for '{operation}'")
                fallback_func = self.fallback_strategies[fallback_operation]
                
                if asyncio.iscoroutinefunction(fallback_func):
                    result = await fallback_func(**kwargs)
                else:
                    result = fallback_func(**kwargs)
                
                FALLBACK_USAGE.labels(fallback_type=fallback_operation).inc()
                return result
                
            except Exception as fallback_error:
                logger.error(f"Fallback failed for '{operation}': {fallback_error}")
                self._add_to_dead_letter_queue(operation, kwargs, str(fallback_error))
        
        # Add to dead letter queue and raise final error
        self._add_to_dead_letter_queue(operation, kwargs, str(last_error))
        raise last_error
    
    def _record_error(self, error_context: ErrorContext):
        """Record error for monitoring and analysis."""
        with self._lock:
            self.error_history[error_context.component].append(error_context)
            
            # Keep only recent errors (last 1000 per component)
            if len(self.error_history[error_context.component]) > 1000:
                self.error_history[error_context.component] = \
                    self.error_history[error_context.component][-1000:]
        
        # Update metrics
        STREAM_ERRORS.labels(
            error_type=error_context.error_type.value,
            component=error_context.component
        ).inc()
        
        logger.error(
            f"Error in {error_context.component}.{error_context.operation}: "
            f"{error_context.error_message}"
        )
    
    def _add_to_dead_letter_queue(self, operation: str, data: Dict[str, Any], error: str):
        """Add failed operation to dead letter queue."""
        dead_letter_item = {
            'operation': operation,
            'data': data,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'retry_count': 0
        }
        
        with self._lock:
            self.dead_letter_queue.append(dead_letter_item)
        
        logger.warning(f"Added to dead letter queue: {operation}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        with self._lock:
            stats = {
                'total_errors': sum(len(errors) for errors in self.error_history.values()),
                'errors_by_component': {comp: len(errors) for comp, errors in self.error_history.items()},
                'dead_letter_queue_size': len(self.dead_letter_queue),
                'circuit_breaker_states': {
                    name: cb.state.value for name, cb in self.circuit_breakers.items()
                }
            }
        
        return stats
    
    def get_dead_letter_items(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get items from dead letter queue."""
        with self._lock:
            return list(self.dead_letter_queue)[-limit:]
    
    def clear_dead_letter_queue(self):
        """Clear dead letter queue."""
        with self._lock:
            self.dead_letter_queue.clear()
        logger.info("Dead letter queue cleared")

# Global error handler instance
stream_error_handler = StreamErrorHandler()

# Convenience functions
def with_stream_error_handling(operation: str, 
                              component: str = "unknown",
                              retry_config: Optional[RetryConfig] = None,
                              circuit_breaker: Optional[str] = None,
                              fallback: Optional[str] = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await stream_error_handler.execute_with_retry(
                func, operation, component, retry_config, circuit_breaker, fallback,
                *args, **kwargs
            )
        return wrapper
    return decorator

def register_stream_circuit_breaker(name: str, 
                                   failure_threshold: int = 5,
                                   timeout: float = 60.0):
    """Register a circuit breaker for a service."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout
    )
    stream_error_handler.register_circuit_breaker(name, config)

def register_stream_fallback(operation: str, fallback_func: Callable):
    """Register a fallback strategy."""
    stream_error_handler.register_fallback(operation, fallback_func)