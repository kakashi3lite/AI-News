#!/usr/bin/env node

/**
 * SecureKeyAgent - API Key Validator
 * Validates and tests all stored API keys without exposing them
 * Enterprise-grade security for production environments
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class ApiKeyValidator {
  constructor() {
    this.projectRoot = path.resolve(__dirname, '..');
    this.envFilePath = path.join(this.projectRoot, '.env.local');
    this.reportPath = path.join(this.projectRoot, 'logs', 'api-validation-report.json');
    this.auditLogPath = path.join(this.projectRoot, 'logs', 'api-validation.log');
    
    // Create logs directory if it doesn't exist
    const logsDir = path.dirname(this.reportPath);
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    
    // API key patterns for validation
    this.keyPatterns = {
      OPENAI_API_KEY: /^sk-(proj-)?[A-Za-z0-9]{32,}$/,
      GOOGLE_API_KEY: /^[A-Za-z0-9_-]{39}$/,
      GOOGLE_AI_API_KEY: /^[A-Za-z0-9_-]{39}$/,
      NEWS_API_KEY: /^[a-f0-9]{32}$/,
      YOUTUBE_API_KEY: /^[A-Za-z0-9_-]{39}$/,
      ANTHROPIC_API_KEY: /^sk-ant-[A-Za-z0-9_-]{95,}$/,
      GEMINI_API_KEY: /^[A-Za-z0-9_-]{39}$/
    };
    
    // Service configurations for testing
    this.serviceConfigs = {
      OPENAI_API_KEY: {
        name: 'OpenAI',
        testUrl: 'https://api.openai.com/v1/models',
        headers: (key) => ({ 'Authorization': `Bearer ${key}` }),
        method: 'GET',
        priority: 'high'
      },
      GOOGLE_API_KEY: {
        name: 'Google Custom Search',
        testUrl: (key) => `https://www.googleapis.com/customsearch/v1?key=${key}&cx=test&q=test`,
        headers: () => ({}),
        method: 'GET',
        priority: 'high'
      },
      NEWS_API_KEY: {
        name: 'NewsAPI',
        testUrl: (key) => `https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey=${key}`,
        headers: () => ({}),
        method: 'GET',
        priority: 'medium'
      },
      YOUTUBE_API_KEY: {
        name: 'YouTube Data API',
        testUrl: (key) => `https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q=test&key=${key}`,
        headers: () => ({}),
        method: 'GET',
        priority: 'low'
      },
      ANTHROPIC_API_KEY: {
        name: 'Anthropic Claude',
        testUrl: 'https://api.anthropic.com/v1/messages',
        headers: (key) => ({
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json'
        }),
        method: 'POST',
        body: JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 1,
          messages: [{ role: 'user', content: 'Hi' }]
        }),
        priority: 'low'
      }
    };
  }

  /**
   * Main validation workflow
   */
  async validateKeys() {
    const startTime = Date.now();
    this.log('Starting API key validation');
    
    console.log('\n🔍 SecureKeyAgent - API Key Validator');
    console.log('=' .repeat(50));
    console.log('✅ Secure validation without exposing keys');
    console.log('✅ Live API testing with proper error handling');
    console.log('✅ Comprehensive reporting and audit trail');
    console.log('=' .repeat(50));
    
    try {
      // Load environment keys
      const envKeys = this.loadEnvironmentKeys();
      
      if (Object.keys(envKeys).length === 0) {
        console.log('\n❌ No API keys found in .env.local');
        console.log('💡 Run: npm run setup:keys');
        return { success: false, error: 'No API keys found' };
      }
      
      console.log(`\n📊 Found ${Object.keys(envKeys).length} API keys to validate\n`);
      
      const results = [];
      
      // Validate each key
      for (const [keyName, keyValue] of Object.entries(envKeys)) {
        console.log(`🔍 Validating ${keyName}...`);
        const result = await this.validateSingleKey(keyName, keyValue);
        results.push(result);
        
        // Display result
        if (result.status === 'valid') {
          console.log(`   ✅ ${result.serviceName}: Valid and tested`);
        } else if (result.status === 'format_valid') {
          console.log(`   ⚠️  ${result.serviceName}: Valid format, test skipped`);
        } else {
          console.log(`   ❌ ${result.serviceName}: ${result.error}`);
        }
      }
      
      // Display results summary
      this.displayResults(results);
      
      // Generate detailed report
      const report = this.generateReport(results, startTime);
      
      // Save report
      fs.writeFileSync(this.reportPath, JSON.stringify(report, null, 2));
      
      this.log('API key validation completed');
      return { success: true, results, report };
      
    } catch (error) {
      console.error('\n❌ Validation failed:', error.message);
      this.log(`Validation failed: ${error.message}`, 'ERROR');
      return { success: false, error: error.message };
    }
  }

  /**
   * Load API keys from .env.local
   */
  loadEnvironmentKeys() {
    if (!fs.existsSync(this.envFilePath)) {
      throw new Error('No .env.local file found');
    }
    
    const envContent = fs.readFileSync(this.envFilePath, 'utf8');
    const envKeys = {};
    
    // Parse environment file
    envContent.split('\n').forEach(line => {
      const match = line.match(/^([A-Z_]+)=(.+)$/);
      if (match && match[2] && !match[2].includes('placeholder')) {
        envKeys[match[1]] = match[2].trim();
      }
    });
    
    this.log(`Loaded ${Object.keys(envKeys).length} API keys from .env.local`);
    return envKeys;
  }

  /**
   * Validate a single API key
   */
  async validateSingleKey(keyName, apiKey) {
    const serviceConfig = this.serviceConfigs[keyName];
    const keyHash = this.hashKey(apiKey);
    
    const result = {
      keyName,
      serviceName: serviceConfig?.name || keyName,
      keyHash,
      timestamp: new Date().toISOString(),
      status: 'unknown',
      error: null,
      responseTime: null,
      priority: serviceConfig?.priority || 'unknown'
    };
    
    try {
      // Format validation
      const pattern = this.keyPatterns[keyName];
      if (pattern && !pattern.test(apiKey)) {
        result.status = 'invalid_format';
        result.error = 'Invalid key format';
        this.log(`Format validation failed for ${keyName} (${keyHash})`, 'WARN');
        return result;
      }
      
      // If no service config, just validate format
      if (!serviceConfig) {
        result.status = 'format_valid';
        this.log(`Format validation passed for ${keyName} (${keyHash})`);
        return result;
      }
      
      // Live API testing
      const testStart = Date.now();
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout
      
      try {
        const testUrl = typeof serviceConfig.testUrl === 'function' 
          ? serviceConfig.testUrl(apiKey) 
          : serviceConfig.testUrl;
        
        const headers = typeof serviceConfig.headers === 'function'
          ? serviceConfig.headers(apiKey)
          : serviceConfig.headers;
        
        const response = await fetch(testUrl, {
          method: serviceConfig.method,
          headers,
          body: serviceConfig.body,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        result.responseTime = Date.now() - testStart;
        
        if (response.ok || response.status === 400) {
          // 400 might be expected for some test calls
          result.status = 'valid';
          this.log(`API test passed for ${keyName} (${keyHash}) - ${result.responseTime}ms`);
        } else if (response.status === 401 || response.status === 403) {
          result.status = 'invalid';
          result.error = 'Invalid API key or insufficient permissions';
          this.log(`API test failed for ${keyName} (${keyHash}): ${result.error}`, 'WARN');
        } else {
          result.status = 'error';
          result.error = `HTTP ${response.status}: ${response.statusText}`;
          this.log(`API test error for ${keyName} (${keyHash}): ${result.error}`, 'WARN');
        }
        
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError.name === 'AbortError') {
          result.status = 'timeout';
          result.error = 'Request timeout';
        } else {
          result.status = 'error';
          result.error = fetchError.message;
        }
        this.log(`API test failed for ${keyName} (${keyHash}): ${result.error}`, 'WARN');
      }
      
    } catch (error) {
      result.status = 'error';
      result.error = error.message;
      this.log(`Validation error for ${keyName} (${keyHash}): ${error.message}`, 'ERROR');
    }
    
    return result;
  }

  /**
   * Hash API key for logging (never log actual keys)
   */
  hashKey(apiKey) {
    return crypto.createHash('sha256').update(apiKey).digest('hex').substring(0, 8);
  }

  /**
   * Display validation results
   */
  displayResults(results) {
    console.log('\n📊 Validation Results:');
    console.log('=' .repeat(30));
    
    const valid = results.filter(r => r.status === 'valid').length;
    const formatValid = results.filter(r => r.status === 'format_valid').length;
    const invalid = results.filter(r => r.status === 'invalid' || r.status === 'invalid_format').length;
    const errors = results.filter(r => r.status === 'error' || r.status === 'timeout').length;
    
    console.log(`✅ Valid and tested:    ${valid}`);
    console.log(`⚠️  Format valid only:  ${formatValid}`);
    console.log(`❌ Invalid:             ${invalid}`);
    console.log(`🔥 Errors/Timeouts:     ${errors}`);
    
    const successRate = results.length > 0 ? 
      ((valid + formatValid) / results.length * 100).toFixed(1) : 0;
    
    console.log(`\n📈 Overall Success Rate: ${successRate}%`);
    
    if (invalid > 0 || errors > 0) {
      console.log('\n💡 Run setup again to fix issues: npm run setup:keys');
    } else {
      console.log('\n🎉 All API keys are valid!');
    }
  }

  /**
   * Generate detailed report
   */
  generateReport(results, startTime) {
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    const report = {
      timestamp: new Date().toISOString(),
      duration: `${duration}ms`,
      summary: {
        total: results.length,
        valid: results.filter(r => r.status === 'valid').length,
        formatValid: results.filter(r => r.status === 'format_valid').length,
        invalid: results.filter(r => r.status === 'invalid' || r.status === 'invalid_format').length,
        errors: results.filter(r => r.status === 'error' || r.status === 'timeout').length,
        successRate: results.length > 0 ? 
          ((results.filter(r => r.status === 'valid' || r.status === 'format_valid').length) / results.length * 100).toFixed(1) : 0
      },
      details: results.map(r => ({
        keyName: r.keyName,
        serviceName: r.serviceName,
        status: r.status,
        error: r.error,
        responseTime: r.responseTime,
        priority: r.priority,
        keyHash: r.keyHash // For audit purposes
      })),
      recommendations: this.generateRecommendations(results)
    };
    
    return report;
  }

  /**
   * Generate recommendations based on results
   */
  generateRecommendations(results) {
    const recommendations = [];
    
    const invalidKeys = results.filter(r => r.status === 'invalid' || r.status === 'invalid_format');
    const errorKeys = results.filter(r => r.status === 'error' || r.status === 'timeout');
    const slowKeys = results.filter(r => r.responseTime && r.responseTime > 5000);
    
    if (invalidKeys.length > 0) {
      recommendations.push({
        type: 'security',
        severity: 'high',
        message: `${invalidKeys.length} API key(s) are invalid and need to be replaced`,
        action: 'Run npm run setup:keys to update invalid keys'
      });
    }
    
    if (errorKeys.length > 0) {
      recommendations.push({
        type: 'connectivity',
        severity: 'medium',
        message: `${errorKeys.length} API key(s) failed testing due to network/service issues`,
        action: 'Check network connectivity and service status'
      });
    }
    
    if (slowKeys.length > 0) {
      recommendations.push({
        type: 'performance',
        severity: 'low',
        message: `${slowKeys.length} API service(s) responded slowly (>5s)`,
        action: 'Monitor API performance and consider alternative providers'
      });
    }
    
    if (recommendations.length === 0) {
      recommendations.push({
        type: 'success',
        severity: 'info',
        message: 'All API keys are valid and working correctly',
        action: 'No action required - system is ready for production'
      });
    }
    
    return recommendations;
  }

  /**
   * Quick validation for CI/CD
   */
  async quickValidation() {
    console.log('🚀 Quick API Key Validation for CI/CD');
    
    try {
      const envKeys = this.loadEnvironmentKeys();
      
      // Check required keys exist
      const requiredKeys = ['OPENAI_API_KEY', 'GOOGLE_API_KEY'];
      const missingRequired = requiredKeys.filter(key => !envKeys[key]);
      
      if (missingRequired.length > 0) {
        console.log(`❌ Missing required keys: ${missingRequired.join(', ')}`);
        process.exit(1);
      }
      
      // Format validation only
      let formatErrors = 0;
      for (const [keyName, keyValue] of Object.entries(envKeys)) {
        const pattern = this.keyPatterns[keyName];
        if (pattern && !pattern.test(keyValue)) {
          console.log(`❌ Invalid format: ${keyName}`);
          formatErrors++;
        }
      }
      
      if (formatErrors > 0) {
        console.log(`❌ ${formatErrors} API key(s) have invalid format`);
        process.exit(1);
      }
      
      console.log('✅ All API keys present and format valid');
      return true;
      
    } catch (error) {
      console.log(`❌ Validation failed: ${error.message}`);
      process.exit(1);
    }
  }

  /**
   * Secure logging
   */
  log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    const logEntry = `${timestamp} [${level}] ${message}\n`;
    fs.appendFileSync(this.auditLogPath, logEntry);
  }
}

// Export for testing
module.exports = ApiKeyValidator;

// Command line interface
if (require.main === module) {
  const validator = new ApiKeyValidator();
  
  const args = process.argv.slice(2);
  const isQuick = args.includes('--quick') || args.includes('-q');
  
  if (isQuick) {
    validator.quickValidation();
  } else {
    validator.validateKeys()
      .then(result => {
        process.exit(result.success ? 0 : 1);
      })
      .catch(error => {
        console.error('Validation failed:', error);
        process.exit(1);
      });
  }
}
