#!/usr/bin/env node

/**
 * SecureKeyAgent - API Key Validator
 * Validates and tests all stored API keys without exposing them
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class ApiKeyValidator {
  constructor() {
    this.projectRoot = path.resolve(__dirname, '..');
    this.envFilePath = path.join(this.projectRoot, '.env.local');
    
    this.services = [
      {
        name: 'OPENAI_API_KEY',
        service: 'OpenAI',
        testEndpoint: 'https://api.openai.com/v1/models',
        testMethod: 'GET',
        headers: (key) => ({ 'Authorization': `Bearer ${key}` }),
        successCodes: [200],
        errorCodes: { 401: 'Invalid API key', 403: 'Insufficient permissions' }
      },
      {
        name: 'GOOGLE_API_KEY',
        service: 'Google Custom Search',
        testEndpoint: (key) => `https://www.googleapis.com/customsearch/v1?key=${key}&cx=test&q=test`,
        testMethod: 'GET',
        headers: () => ({}),
        successCodes: [200, 400], // 400 is expected for invalid cx
        errorCodes: { 403: 'API key invalid or quota exceeded' }
      },
      {
        name: 'NEWS_API_KEY',
        service: 'NewsAPI',
        testEndpoint: (key) => `https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey=${key}`,
        testMethod: 'GET',
        headers: () => ({}),
        successCodes: [200],
        errorCodes: { 401: 'Invalid API key', 429: 'Rate limit exceeded' }
      },
      {
        name: 'YOUTUBE_API_KEY',
        service: 'YouTube Data API',
        testEndpoint: (key) => `https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q=test&key=${key}`,
        testMethod: 'GET',
        headers: () => ({}),
        successCodes: [200],
        errorCodes: { 403: 'API key invalid or quota exceeded' }
      },
      {
        name: 'ANTHROPIC_API_KEY',
        service: 'Anthropic Claude',
        testEndpoint: 'https://api.anthropic.com/v1/messages',
        testMethod: 'POST',
        headers: (key) => ({
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json'
        }),
        body: {
          model: 'claude-3-haiku-20240307',
          max_tokens: 1,
          messages: [{ role: 'user', content: 'Hi' }]
        },
        successCodes: [200, 400], // 400 might be expected for minimal request
        errorCodes: { 401: 'Invalid API key', 403: 'Insufficient permissions' }
      }
    ];
  }

  /**
   * Main validation workflow
   */
  async validateKeys() {
    console.log('\n🔍 SecureKeyAgent - API Key Validator');
    console.log('=' .repeat(50));
    
    try {
      const envKeys = this.loadEnvironmentKeys();
      
      if (Object.keys(envKeys).length === 0) {
        console.log('❌ No API keys found in .env.local');
        console.log('💡 Run: npm run setup:keys');
        process.exit(1);
      }
      
      console.log(`📊 Found ${Object.keys(envKeys).length} API keys to validate\n`);
      
      const results = [];
      
      for (const service of this.services) {
        if (envKeys[service.name]) {
          const result = await this.validateSingleKey(service, envKeys[service.name]);
          results.push(result);
        } else {
          results.push({
            service: service.service,
            key: service.name,
            status: 'missing',
            message: 'Key not configured'
          });
        }
      }
      
      this.displayResults(results);
      this.generateReport(results);
      
    } catch (error) {
      console.error('❌ Validation failed:', error.message);
      process.exit(1);
    }
  }

  /**
   * Load API keys from .env.local
   */
  loadEnvironmentKeys() {
    if (!fs.existsSync(this.envFilePath)) {
      return {};
    }
    
    const envContent = fs.readFileSync(this.envFilePath, 'utf8');
    const envKeys = {};
    
    envContent.split('\n').forEach(line => {
      const match = line.match(/^([A-Z_]+)=(.+)$/);
      if (match && match[1].includes('API_KEY')) {
        envKeys[match[1]] = match[2].trim();
      }
    });
    
    return envKeys;
  }

  /**
   * Validate a single API key
   */
  async validateSingleKey(service, apiKey) {
    const keyHash = this.hashKey(apiKey);
    
    console.log(`🔍 Testing ${service.service}...`);
    
    try {
      const testUrl = typeof service.testEndpoint === 'function' 
        ? service.testEndpoint(apiKey) 
        : service.testEndpoint;
        
      const headers = service.headers(apiKey);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);
      
      const response = await fetch(testUrl, {
        method: service.testMethod,
        headers,
        body: service.body ? JSON.stringify(service.body) : undefined,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (service.successCodes.includes(response.status)) {
        console.log(`   ✅ ${service.service}: Valid`);
        return {
          service: service.service,
          key: service.name,
          keyHash,
          status: 'valid',
          message: 'API key validated successfully',
          responseCode: response.status
        };
      } else if (service.errorCodes[response.status]) {
        console.log(`   ❌ ${service.service}: ${service.errorCodes[response.status]}`);
        return {
          service: service.service,
          key: service.name,
          keyHash,
          status: 'invalid',
          message: service.errorCodes[response.status],
          responseCode: response.status
        };
      } else {
        console.log(`   ⚠️  ${service.service}: Unexpected response (${response.status})`);
        return {
          service: service.service,
          key: service.name,
          keyHash,
          status: 'warning',
          message: `Unexpected HTTP ${response.status}`,
          responseCode: response.status
        };
      }
      
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log(`   ⏱️  ${service.service}: Timeout`);
        return {
          service: service.service,
          key: service.name,
          keyHash,
          status: 'timeout',
          message: 'Request timeout - check internet connection'
        };
      } else {
        console.log(`   ❌ ${service.service}: ${error.message}`);
        return {
          service: service.service,
          key: service.name,
          keyHash,
          status: 'error',
          message: error.message
        };
      }
    }
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
    console.log('=' .repeat(50));
    
    const valid = results.filter(r => r.status === 'valid').length;
    const invalid = results.filter(r => r.status === 'invalid').length;
    const missing = results.filter(r => r.status === 'missing').length;
    const errors = results.filter(r => ['error', 'timeout', 'warning'].includes(r.status)).length;
    
    console.log(`✅ Valid:   ${valid}`);
    console.log(`❌ Invalid: ${invalid}`);
    console.log(`⚠️  Missing: ${missing}`);
    console.log(`🔄 Errors:  ${errors}`);
    
    if (invalid > 0 || missing > 0) {
      console.log('\n🔧 Action Required:');
      results.forEach(result => {
        if (result.status === 'invalid' || result.status === 'missing') {
          console.log(`   • ${result.service}: ${result.message}`);
        }
      });
      console.log('\n💡 Run: npm run setup:keys');
    } else {
      console.log('\n🎉 All API keys are valid!');
    }
  }

  /**
   * Generate detailed report
   */
  generateReport(results) {
    const reportPath = path.join(this.projectRoot, 'api-key-validation-report.json');
    
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total: results.length,
        valid: results.filter(r => r.status === 'valid').length,
        invalid: results.filter(r => r.status === 'invalid').length,
        missing: results.filter(r => r.status === 'missing').length,
        errors: results.filter(r => ['error', 'timeout', 'warning'].includes(r.status)).length
      },
      results: results.map(r => ({
        service: r.service,
        key: r.key,
        keyHash: r.keyHash || 'N/A',
        status: r.status,
        message: r.message,
        responseCode: r.responseCode
      })),
      security: {
        envFileExists: fs.existsSync(this.envFilePath),
        envFilePermissions: fs.existsSync(this.envFilePath) ? 
          (fs.statSync(this.envFilePath).mode & parseInt('777', 8)).toString(8) : 'N/A'
      }
    };
    
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Detailed report saved: ${reportPath}`);
  }

  /**
   * Quick validation for CI/CD
   */
  async quickValidation() {
    const envKeys = this.loadEnvironmentKeys();
    const requiredKeys = ['OPENAI_API_KEY', 'GOOGLE_API_KEY'];
    
    const missing = requiredKeys.filter(key => !envKeys[key]);
    
    if (missing.length > 0) {
      console.log(`❌ Missing required keys: ${missing.join(', ')}`);
      process.exit(1);
    }
    
    console.log('✅ All required keys present');
    process.exit(0);
  }
}

// Export for testing
module.exports = ApiKeyValidator;

// Command line interface
if (require.main === module) {
  const validator = new ApiKeyValidator();
  
  const command = process.argv[2];
  
  switch (command) {
    case 'quick':
      validator.quickValidation();
      break;
    case 'full':
    default:
      validator.validateKeys();
      break;
  }
}
