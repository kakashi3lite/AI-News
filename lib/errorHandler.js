/**
 * Error handling utilities with retry and circuit breaker support.
 */

const ERROR_TYPES = {
  NETWORK: 'network',
  TIMEOUT: 'timeout',
  RATE_LIMIT: 'rate_limit',
  AUTHENTICATION: 'authentication',
  VALIDATION: 'validation',
  PARSING: 'parsing',
  EXTERNAL_API: 'external_api',
  INTERNAL: 'internal'
};

const classifyError = (error) => {
  if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') return ERROR_TYPES.NETWORK;
  if (error.code === 'ETIMEDOUT' || error.message?.includes('timeout')) return ERROR_TYPES.TIMEOUT;
  if (error.response?.status === 429) return ERROR_TYPES.RATE_LIMIT;
  if (error.response?.status === 401 || error.response?.status === 403) return ERROR_TYPES.AUTHENTICATION;
  if (error.response?.status >= 400 && error.response?.status < 500) return ERROR_TYPES.VALIDATION;
  if (error.response?.status >= 500) return ERROR_TYPES.EXTERNAL_API;
  if (error.message?.includes('parse') || error.message?.includes('JSON')) return ERROR_TYPES.PARSING;
  return ERROR_TYPES.INTERNAL;
};

class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000;
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.lastFailureTime = null;
  }

  async execute(operation, fallback = null) {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = 'HALF_OPEN';
        this.failureCount = 0;
      } else if (fallback) {
        return await fallback();
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await operation();
      this.failureCount = 0;
      if (this.state === 'HALF_OPEN') this.state = 'CLOSED';
      return result;
    } catch (error) {
      this.failureCount++;
      this.lastFailureTime = Date.now();
      if (this.failureCount >= this.failureThreshold) this.state = 'OPEN';
      throw error;
    }
  }
}

class RetryManager {
  constructor(options = {}) {
    this.maxRetries = options.maxRetries || 3;
    this.baseDelay = options.baseDelay || 1000;
    this.maxDelay = options.maxDelay || 30000;
    this.backoffMultiplier = options.backoffMultiplier || 2;
  }

  async executeWithRetry(operation, options = {}) {
    const maxRetries = options.maxRetries || this.maxRetries;
    const retryableErrors = options.retryableErrors || [ERROR_TYPES.NETWORK, ERROR_TYPES.TIMEOUT, ERROR_TYPES.RATE_LIMIT];
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await operation(attempt);
      } catch (error) {
        lastError = error;
        const errorType = classifyError(error);
        if (attempt === maxRetries || !retryableErrors.includes(errorType)) throw error;

        const baseDelay = this.baseDelay * Math.pow(this.backoffMultiplier, attempt);
        const jitteredDelay = Math.floor(Math.min(baseDelay, this.maxDelay) * (0.5 + Math.random() * 0.5));
        await new Promise((resolve) => setTimeout(resolve, jitteredDelay));
      }
    }
    throw lastError;
  }
}

class ErrorHandler {
  constructor(options = {}) {
    this.retryManager = new RetryManager(options.retry);
    this.circuitBreakers = new Map();
  }

  async executeWithErrorHandling(operation, options = {}) {
    const { name = 'unknown', retryOptions = {}, circuitBreakerOptions = {}, fallback = null, useCircuitBreaker = true, useRetry = true } = options;

    try {
      if (useCircuitBreaker) {
        if (!this.circuitBreakers.has(name)) {
          this.circuitBreakers.set(name, new CircuitBreaker(circuitBreakerOptions));
        }
        const cb = this.circuitBreakers.get(name);
        return await cb.execute(
          useRetry ? () => this.retryManager.executeWithRetry(operation, retryOptions) : operation,
          fallback
        );
      }

      if (useRetry) return await this.retryManager.executeWithRetry(operation, retryOptions);
      return await operation();
    } catch (error) {
      if (fallback && !useCircuitBreaker) {
        try { return await fallback(); } catch { throw error; }
      }
      throw error;
    }
  }
}

const errorHandler = new ErrorHandler();

module.exports = { ErrorHandler, CircuitBreaker, RetryManager, ERROR_TYPES, classifyError, errorHandler };
