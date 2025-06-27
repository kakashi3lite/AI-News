/**
 * Enhanced Error Handling Utility
 * Dr. NewsForge's AI News Dashboard
 * 
 * Features:
 * - Retry mechanisms with exponential backoff
 * - Circuit breaker pattern
 * - Error classification and recovery strategies
 * - Monitoring and alerting integration
 * - Graceful degradation
 */

const EventEmitter = require('events');

// Error types for classification
const ERROR_TYPES = {
  NETWORK: 'network',
  TIMEOUT: 'timeout',
  RATE_LIMIT: 'rate_limit',
  AUTHENTICATION: 'authentication',
  VALIDATION: 'validation',
  PARSING: 'parsing',
  RESOURCE: 'resource',
  EXTERNAL_API: 'external_api',
  INTERNAL: 'internal'
};

// Recovery strategies
const RECOVERY_STRATEGIES = {
  RETRY: 'retry',
  FALLBACK: 'fallback',
  SKIP: 'skip',
  ALERT: 'alert',
  CIRCUIT_BREAK: 'circuit_break'
};

class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000; // 1 minute
    this.monitoringPeriod = options.monitoringPeriod || 10000; // 10 seconds
    
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failureCount = 0;
    this.lastFailureTime = null;
    this.successCount = 0;
    this.requestCount = 0;
  }

  async execute(operation, fallback = null) {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = 'HALF_OPEN';
        this.failureCount = 0;
      } else {
        if (fallback) {
          console.log('🔴 Circuit breaker OPEN - using fallback');
          return await fallback();
        }
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failureCount = 0;
    this.successCount++;
    
    if (this.state === 'HALF_OPEN') {
      this.state = 'CLOSED';
      console.log('🟢 Circuit breaker CLOSED - service recovered');
    }
  }

  onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      console.log('🔴 Circuit breaker OPEN - too many failures');
    }
  }

  getStats() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime
    };
  }
}

class RetryManager {
  constructor(options = {}) {
    this.maxRetries = options.maxRetries || 3;
    this.baseDelay = options.baseDelay || 1000;
    this.maxDelay = options.maxDelay || 30000;
    this.backoffMultiplier = options.backoffMultiplier || 2;
    this.jitter = options.jitter || true;
  }

  async executeWithRetry(operation, options = {}) {
    const maxRetries = options.maxRetries || this.maxRetries;
    const retryableErrors = options.retryableErrors || [ERROR_TYPES.NETWORK, ERROR_TYPES.TIMEOUT, ERROR_TYPES.RATE_LIMIT];
    
    let lastError;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const result = await operation(attempt);
        if (attempt > 0) {
          console.log(`✅ Operation succeeded on attempt ${attempt + 1}`);
        }
        return result;
      } catch (error) {
        lastError = error;
        const errorType = this.classifyError(error);
        
        if (attempt === maxRetries || !retryableErrors.includes(errorType)) {
          console.log(`❌ Operation failed after ${attempt + 1} attempts: ${error.message}`);
          throw error;
        }
        
        const delay = this.calculateDelay(attempt);
        console.log(`⚠️ Attempt ${attempt + 1} failed (${errorType}), retrying in ${delay}ms...`);
        await this.delay(delay);
      }
    }
    
    throw lastError;
  }

  calculateDelay(attempt) {
    let delay = this.baseDelay * Math.pow(this.backoffMultiplier, attempt);
    delay = Math.min(delay, this.maxDelay);
    
    if (this.jitter) {
      delay = delay * (0.5 + Math.random() * 0.5); // Add 0-50% jitter
    }
    
    return Math.floor(delay);
  }

  classifyError(error) {
    if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
      return ERROR_TYPES.NETWORK;
    }
    if (error.code === 'ETIMEDOUT' || error.message.includes('timeout')) {
      return ERROR_TYPES.TIMEOUT;
    }
    if (error.response?.status === 429) {
      return ERROR_TYPES.RATE_LIMIT;
    }
    if (error.response?.status === 401 || error.response?.status === 403) {
      return ERROR_TYPES.AUTHENTICATION;
    }
    if (error.response?.status >= 400 && error.response?.status < 500) {
      return ERROR_TYPES.VALIDATION;
    }
    if (error.response?.status >= 500) {
      return ERROR_TYPES.EXTERNAL_API;
    }
    if (error.message.includes('parse') || error.message.includes('JSON')) {
      return ERROR_TYPES.PARSING;
    }
    return ERROR_TYPES.INTERNAL;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class ErrorHandler extends EventEmitter {
  constructor(options = {}) {
    super();
    this.retryManager = new RetryManager(options.retry);
    this.circuitBreakers = new Map();
    this.errorStats = new Map();
    this.alertThresholds = options.alertThresholds || {
      errorRate: 0.1, // 10%
      timeWindow: 300000 // 5 minutes
    };
    this.monitoringEnabled = options.monitoring !== false;
    
    if (this.monitoringEnabled) {
      this.startMonitoring();
    }
  }

  getCircuitBreaker(name, options = {}) {
    if (!this.circuitBreakers.has(name)) {
      this.circuitBreakers.set(name, new CircuitBreaker(options));
    }
    return this.circuitBreakers.get(name);
  }

  async executeWithErrorHandling(operation, options = {}) {
    const {
      name = 'unknown',
      retryOptions = {},
      circuitBreakerOptions = {},
      fallback = null,
      useCircuitBreaker = true,
      useRetry = true
    } = options;

    const startTime = Date.now();
    
    try {
      let result;
      
      if (useCircuitBreaker) {
        const circuitBreaker = this.getCircuitBreaker(name, circuitBreakerOptions);
        
        if (useRetry) {
          result = await circuitBreaker.execute(async () => {
            return await this.retryManager.executeWithRetry(operation, retryOptions);
          }, fallback);
        } else {
          result = await circuitBreaker.execute(operation, fallback);
        }
      } else if (useRetry) {
        result = await this.retryManager.executeWithRetry(operation, retryOptions);
      } else {
        result = await operation();
      }
      
      this.recordSuccess(name, Date.now() - startTime);
      return result;
      
    } catch (error) {
      this.recordError(name, error, Date.now() - startTime);
      
      if (fallback && !useCircuitBreaker) {
        console.log(`🔄 Using fallback for ${name}`);
        try {
          return await fallback();
        } catch (fallbackError) {
          console.error(`❌ Fallback also failed for ${name}:`, fallbackError.message);
          throw error; // Throw original error
        }
      }
      
      throw error;
    }
  }

  recordSuccess(operationName, duration) {
    if (!this.errorStats.has(operationName)) {
      this.errorStats.set(operationName, {
        successes: 0,
        errors: 0,
        totalDuration: 0,
        recentErrors: []
      });
    }
    
    const stats = this.errorStats.get(operationName);
    stats.successes++;
    stats.totalDuration += duration;
    
    this.emit('success', { operationName, duration });
  }

  recordError(operationName, error, duration) {
    if (!this.errorStats.has(operationName)) {
      this.errorStats.set(operationName, {
        successes: 0,
        errors: 0,
        totalDuration: 0,
        recentErrors: []
      });
    }
    
    const stats = this.errorStats.get(operationName);
    stats.errors++;
    stats.totalDuration += duration;
    stats.recentErrors.push({
      timestamp: Date.now(),
      error: error.message,
      type: this.retryManager.classifyError(error)
    });
    
    // Keep only recent errors (last 5 minutes)
    const fiveMinutesAgo = Date.now() - 300000;
    stats.recentErrors = stats.recentErrors.filter(e => e.timestamp > fiveMinutesAgo);
    
    this.emit('error', { operationName, error, duration });
    this.checkAlertThresholds(operationName, stats);
  }

  checkAlertThresholds(operationName, stats) {
    const total = stats.successes + stats.errors;
    if (total < 10) return; // Need minimum sample size
    
    const errorRate = stats.errors / total;
    if (errorRate > this.alertThresholds.errorRate) {
      this.emit('alert', {
        type: 'high_error_rate',
        operationName,
        errorRate,
        threshold: this.alertThresholds.errorRate,
        stats
      });
    }
  }

  getStats(operationName = null) {
    if (operationName) {
      const stats = this.errorStats.get(operationName);
      if (!stats) return null;
      
      const total = stats.successes + stats.errors;
      return {
        ...stats,
        total,
        errorRate: total > 0 ? stats.errors / total : 0,
        avgDuration: total > 0 ? stats.totalDuration / total : 0
      };
    }
    
    const allStats = {};
    for (const [name, stats] of this.errorStats) {
      allStats[name] = this.getStats(name);
    }
    return allStats;
  }

  getCircuitBreakerStats() {
    const stats = {};
    for (const [name, breaker] of this.circuitBreakers) {
      stats[name] = breaker.getStats();
    }
    return stats;
  }

  startMonitoring() {
    setInterval(() => {
      const stats = this.getStats();
      const circuitStats = this.getCircuitBreakerStats();
      
      this.emit('monitoring', {
        timestamp: Date.now(),
        errorStats: stats,
        circuitBreakerStats: circuitStats
      });
    }, 60000); // Every minute
  }

  // Graceful degradation helpers
  createFallbackFunction(primaryOperation, fallbackOperation, options = {}) {
    return async (...args) => {
      try {
        return await this.executeWithErrorHandling(
          () => primaryOperation(...args),
          {
            ...options,
            fallback: () => fallbackOperation(...args)
          }
        );
      } catch (error) {
        console.warn(`⚠️ Both primary and fallback operations failed: ${error.message}`);
        throw error;
      }
    };
  }
}

// Export singleton instance and classes
const errorHandler = new ErrorHandler();

module.exports = {
  ErrorHandler,
  CircuitBreaker,
  RetryManager,
  ERROR_TYPES,
  RECOVERY_STRATEGIES,
  errorHandler // Singleton instance
};