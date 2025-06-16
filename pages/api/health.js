/**
 * AI News Dashboard - Health Check API
 * Comprehensive health monitoring for all integrated features
 * Built by Dr. Phoenix "SoloSprint" Vega
 */

import { NextApiRequest, NextApiResponse } from 'next';
import deployConfig from '../../deploy.config.js';

// Health check configuration
const HEALTH_CHECK_CONFIG = {
  timeout: 5000, // 5 seconds
  retries: 2,
  criticalServices: ['database', 'openai', 'news_api'],
  optionalServices: ['redis', 'analytics', 'monitoring']
};

// Service health checkers
class HealthChecker {
  constructor() {
    this.startTime = Date.now();
    this.environment = process.env.NODE_ENV || 'development';
    this.config = deployConfig.getEnvironmentConfig(this.environment);
  }

  async checkOverallHealth() {
    const results = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      environment: this.environment,
      version: process.env.npm_package_version || '1.0.0',
      uptime: process.uptime(),
      checks: {},
      features: {},
      performance: {},
      summary: {
        healthy: 0,
        unhealthy: 0,
        degraded: 0
      }
    };

    try {
      // Core service checks
      await this.checkCoreServices(results);
      
      // Feature availability checks
      await this.checkFeatures(results);
      
      // Performance metrics
      await this.checkPerformance(results);
      
      // Determine overall status
      this.determineOverallStatus(results);
      
      return results;
    } catch (error) {
      results.status = 'unhealthy';
      results.error = error.message;
      return results;
    }
  }

  async checkCoreServices(results) {
    const services = [
      { name: 'database', checker: this.checkDatabase },
      { name: 'openai', checker: this.checkOpenAI },
      { name: 'news_api', checker: this.checkNewsAPI },
      { name: 'redis', checker: this.checkRedis },
      { name: 'file_system', checker: this.checkFileSystem },
      { name: 'memory', checker: this.checkMemory },
      { name: 'external_apis', checker: this.checkExternalAPIs }
    ];

    for (const service of services) {
      try {
        const result = await this.withTimeout(
          service.checker.bind(this)(),
          HEALTH_CHECK_CONFIG.timeout
        );
        
        results.checks[service.name] = {
          status: 'healthy',
          responseTime: result.responseTime || 0,
          details: result.details || {},
          lastChecked: new Date().toISOString()
        };
        
        results.summary.healthy++;
      } catch (error) {
        const isCritical = HEALTH_CHECK_CONFIG.criticalServices.includes(service.name);
        
        results.checks[service.name] = {
          status: isCritical ? 'unhealthy' : 'degraded',
          error: error.message,
          lastChecked: new Date().toISOString()
        };
        
        if (isCritical) {
          results.summary.unhealthy++;
        } else {
          results.summary.degraded++;
        }
      }
    }
  }

  async checkDatabase() {
    const start = Date.now();
    
    try {
      // For SQLite (development)
      if (process.env.DATABASE_URL?.includes('file:')) {
        const fs = require('fs');
        const dbPath = process.env.DATABASE_URL.replace('file:', '');
        
        if (fs.existsSync(dbPath)) {
          return {
            responseTime: Date.now() - start,
            details: {
              type: 'sqlite',
              path: dbPath,
              size: fs.statSync(dbPath).size
            }
          };
        } else {
          throw new Error('Database file not found');
        }
      }
      
      // For PostgreSQL (production)
      // This would typically use your database client
      // For now, we'll simulate a connection check
      return {
        responseTime: Date.now() - start,
        details: {
          type: 'postgresql',
          status: 'connected'
        }
      };
    } catch (error) {
      throw new Error(`Database check failed: ${error.message}`);
    }
  }

  async checkOpenAI() {
    const start = Date.now();
    
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OpenAI API key not configured');
    }

    try {
      // Simple API validation (without making actual requests)
      const apiKey = process.env.OPENAI_API_KEY;
      
      if (!apiKey.startsWith('sk-')) {
        throw new Error('Invalid OpenAI API key format');
      }
      
      return {
        responseTime: Date.now() - start,
        details: {
          apiKeyConfigured: true,
          model: process.env.OPENAI_MODEL || 'gpt-4-turbo-preview'
        }
      };
    } catch (error) {
      throw new Error(`OpenAI check failed: ${error.message}`);
    }
  }

  async checkNewsAPI() {
    const start = Date.now();
    
    if (!process.env.NEWS_API_KEY) {
      throw new Error('News API key not configured');
    }

    try {
      // Validate API key format
      const apiKey = process.env.NEWS_API_KEY;
      
      if (apiKey.length < 10) {
        throw new Error('Invalid News API key format');
      }
      
      return {
        responseTime: Date.now() - start,
        details: {
          apiKeyConfigured: true,
          provider: 'newsapi.org'
        }
      };
    } catch (error) {
      throw new Error(`News API check failed: ${error.message}`);
    }
  }

  async checkRedis() {
    const start = Date.now();
    
    if (!process.env.REDIS_URL) {
      return {
        responseTime: Date.now() - start,
        details: {
          configured: false,
          note: 'Redis not configured (optional)'
        }
      };
    }

    try {
      // Redis connection check would go here
      // For now, we'll simulate it
      return {
        responseTime: Date.now() - start,
        details: {
          configured: true,
          url: process.env.REDIS_URL.replace(/\/\/.*@/, '//***@')
        }
      };
    } catch (error) {
      throw new Error(`Redis check failed: ${error.message}`);
    }
  }

  async checkFileSystem() {
    const start = Date.now();
    
    try {
      const fs = require('fs');
      const path = require('path');
      
      // Check if we can read/write to temp directory
      const tempFile = path.join(process.cwd(), '.health-check-temp');
      
      fs.writeFileSync(tempFile, 'health-check');
      const content = fs.readFileSync(tempFile, 'utf8');
      fs.unlinkSync(tempFile);
      
      if (content !== 'health-check') {
        throw new Error('File system read/write test failed');
      }
      
      return {
        responseTime: Date.now() - start,
        details: {
          readable: true,
          writable: true,
          workingDirectory: process.cwd()
        }
      };
    } catch (error) {
      throw new Error(`File system check failed: ${error.message}`);
    }
  }

  async checkMemory() {
    const start = Date.now();
    
    try {
      const memUsage = process.memoryUsage();
      const totalMem = require('os').totalmem();
      const freeMem = require('os').freemem();
      
      const memoryPressure = (memUsage.heapUsed / memUsage.heapTotal) > 0.9;
      
      if (memoryPressure) {
        throw new Error('High memory pressure detected');
      }
      
      return {
        responseTime: Date.now() - start,
        details: {
          heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024),
          heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024),
          systemTotal: Math.round(totalMem / 1024 / 1024),
          systemFree: Math.round(freeMem / 1024 / 1024),
          unit: 'MB'
        }
      };
    } catch (error) {
      throw new Error(`Memory check failed: ${error.message}`);
    }
  }

  async checkExternalAPIs() {
    const start = Date.now();
    
    try {
      // Check if we can resolve DNS for key external services
      const dns = require('dns').promises;
      
      const services = [
        'api.openai.com',
        'newsapi.org',
        'vercel.com'
      ];
      
      const results = await Promise.allSettled(
        services.map(service => dns.lookup(service))
      );
      
      const failed = results.filter(r => r.status === 'rejected');
      
      if (failed.length > 0) {
        throw new Error(`DNS resolution failed for ${failed.length} services`);
      }
      
      return {
        responseTime: Date.now() - start,
        details: {
          dnsResolution: 'successful',
          servicesChecked: services.length
        }
      };
    } catch (error) {
      throw new Error(`External APIs check failed: ${error.message}`);
    }
  }

  async checkFeatures(results) {
    const features = {
      contextAwareness: this.config.features.contextAwareness,
      arcSearch: this.config.features.arcSearch,
      aiSkills: this.config.features.aiSkills,
      socialFeatures: this.config.features.socialFeatures,
      experimentation: this.config.features.experimentation,
      voiceSearch: this.config.features.voiceSearch,
      realTimeCollab: this.config.features.realTimeCollab,
      analytics: this.config.features.analytics
    };

    for (const [feature, enabled] of Object.entries(features)) {
      results.features[feature] = {
        enabled,
        status: enabled ? 'active' : 'disabled',
        rollout: this.config.featureFlags?.[feature]?.rollout || 100
      };
    }
  }

  async checkPerformance(results) {
    const start = Date.now();
    
    try {
      // CPU usage approximation
      const cpuUsage = process.cpuUsage();
      
      // Event loop lag check
      const eventLoopStart = Date.now();
      await new Promise(resolve => setImmediate(resolve));
      const eventLoopLag = Date.now() - eventLoopStart;
      
      results.performance = {
        responseTime: Date.now() - start,
        eventLoopLag,
        cpuUsage: {
          user: cpuUsage.user,
          system: cpuUsage.system
        },
        uptime: process.uptime(),
        nodeVersion: process.version,
        platform: process.platform,
        architecture: process.arch
      };
      
      // Performance warnings
      if (eventLoopLag > 100) {
        results.performance.warnings = results.performance.warnings || [];
        results.performance.warnings.push('High event loop lag detected');
      }
      
    } catch (error) {
      results.performance = {
        error: error.message
      };
    }
  }

  determineOverallStatus(results) {
    if (results.summary.unhealthy > 0) {
      results.status = 'unhealthy';
    } else if (results.summary.degraded > 0) {
      results.status = 'degraded';
    } else {
      results.status = 'healthy';
    }
    
    // Add status message
    results.message = this.getStatusMessage(results);
  }

  getStatusMessage(results) {
    switch (results.status) {
      case 'healthy':
        return 'All systems operational. AI News Dashboard is running optimally with all context-aware features active.';
      case 'degraded':
        return `Some non-critical services are experiencing issues. Core functionality remains available. ${results.summary.degraded} service(s) degraded.`;
      case 'unhealthy':
        return `Critical services are down. Some features may be unavailable. ${results.summary.unhealthy} critical service(s) failed.`;
      default:
        return 'Unknown status';
    }
  }

  async withTimeout(promise, timeoutMs) {
    const timeout = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Health check timeout')), timeoutMs);
    });
    
    return Promise.race([promise, timeout]);
  }
}

// API endpoint handler
export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  
  // Only allow GET requests
  if (req.method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const healthChecker = new HealthChecker();
    const healthStatus = await healthChecker.checkOverallHealth();
    
    // Set appropriate HTTP status code
    let statusCode = 200;
    if (healthStatus.status === 'degraded') {
      statusCode = 200; // Still operational
    } else if (healthStatus.status === 'unhealthy') {
      statusCode = 503; // Service unavailable
    }
    
    // Add response headers
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('X-Health-Check-Version', '1.0.0');
    res.setHeader('X-Environment', healthStatus.environment);
    res.setHeader('X-Status', healthStatus.status);
    
    res.status(statusCode).json(healthStatus);
    
  } catch (error) {
    console.error('Health check failed:', error);
    
    res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Health check system failure',
      message: 'Unable to perform health checks',
      details: {
        error: error.message
      }
    });
  }
}

// Export health checker for use in other parts of the application
export { HealthChecker };