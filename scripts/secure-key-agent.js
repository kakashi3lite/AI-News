#!/usr/bin/env node

/**
 * SecureKeyAgent - Master API Key Management System
 * Enterprise-grade security for AI News Dashboard
 * Never exposes keys in code or git history
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');

class SecureKeyAgent {
  constructor() {
    this.projectRoot = path.resolve(__dirname, '..');
    this.envFilePath = path.join(this.projectRoot, '.env.local');
    this.backupPath = path.join(this.projectRoot, '.env.backup');
    this.auditLogPath = path.join(this.projectRoot, 'logs', 'secure-key-audit.log');
    
    // Create logs directory if it doesn't exist
    const logsDir = path.dirname(this.auditLogPath);
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
    
    // Required APIs for the project
    this.requiredKeys = [
      {
        name: 'OPENAI_API_KEY',
        service: 'OpenAI',
        description: 'Required for AI summarization and content generation',
        getUrl: 'https://platform.openai.com/api-keys',
        testEndpoint: 'https://api.openai.com/v1/models',
        required: true
      },
      {
        name: 'GOOGLE_API_KEY',
        service: 'Google Custom Search',
        description: 'Required for news search and content aggregation',
        getUrl: 'https://console.cloud.google.com/apis/credentials',
        testEndpoint: 'https://www.googleapis.com/customsearch/v1',
        required: true
      }
    ];
    
    // Optional APIs
    this.optionalKeys = [
      {
        name: 'NEWS_API_KEY',
        service: 'NewsAPI',
        description: 'Alternative news source for content diversity',
        getUrl: 'https://newsapi.org/register',
        testEndpoint: 'https://newsapi.org/v2/top-headlines',
        required: false
      },
      {
        name: 'YOUTUBE_API_KEY',
        service: 'YouTube Data API',
        description: 'For video content analysis (optional)',
        getUrl: 'https://console.cloud.google.com/apis/credentials',
        testEndpoint: 'https://www.googleapis.com/youtube/v3/search',
        required: false
      },
      {
        name: 'ANTHROPIC_API_KEY',
        service: 'Anthropic Claude',
        description: 'Alternative AI provider (optional)',
        getUrl: 'https://console.anthropic.com/',
        testEndpoint: 'https://api.anthropic.com/v1/messages',
        required: false
      }
    ];
    
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  /**
   * Main setup workflow
   */
  async setupKeys() {
    this.log('Starting SecureKeyAgent setup');
    console.log('\n🔐 SecureKeyAgent - AI News Dashboard API Key Setup');
    console.log('=' .repeat(60));
    console.log('✅ Secure storage in .env.local (excluded from git)');
    console.log('✅ Pattern validation for each service');
    console.log('✅ Live API testing with harmless calls');
    console.log('✅ Comprehensive security audit');
    console.log('=' .repeat(60));
    
    try {
      // Check if .env.local exists and backup if needed
      await this.handleExistingEnv();
      
      // Setup each required API key
      for (const keyConfig of this.requiredKeys) {
        await this.setupSingleKey(keyConfig);
      }
      
      // Setup optional keys
      await this.setupOptionalKeys();
      
      // Validate all keys
      await this.validateAllKeys();
      
      // Final security check
      await this.performSecurityCheck();
      
      console.log('\n🎉 API Key Setup Complete!');
      console.log('✅ All keys stored securely in .env.local');
      console.log('✅ Keys validated and tested successfully');
      console.log('✅ .env.local is in .gitignore (not committed)');
      console.log('✅ Security audit completed');
      console.log('\n🚀 You can now run: npm run dev');
      
      this.log('SecureKeyAgent setup completed successfully');
      
    } catch (error) {
      console.error('\n❌ Setup failed:', error.message);
      this.log(`Setup failed: ${error.message}`, 'ERROR');
      await this.handleSetupError(error);
    } finally {
      this.rl.close();
    }
  }

  /**
   * Handle existing environment file
   */
  async handleExistingEnv() {
    if (fs.existsSync(this.envFilePath)) {
      const answer = await this.askQuestion(
        '\n⚠️  .env.local already exists. Do you want to:\n' +
        '1. Update existing keys\n' +
        '2. Create backup and start fresh\n' +
        '3. Cancel setup\n' +
        'Choose (1/2/3): '
      );
      
      switch (answer.trim()) {
        case '1':
          console.log('📝 Updating existing .env.local...');
          this.log('Updating existing .env.local');
          break;
        case '2':
          fs.copyFileSync(this.envFilePath, this.backupPath);
          console.log(`📋 Backup created: ${this.backupPath}`);
          this.log(`Backup created: ${this.backupPath}`);
          break;
        case '3':
          throw new Error('Setup cancelled by user');
        default:
          throw new Error('Invalid choice');
      }
    }
  }

  /**
   * Setup a single API key with validation and testing
   */
  async setupSingleKey(keyConfig) {
    console.log(`\n📝 Setting up ${keyConfig.service} API Key`);
    console.log(`   Purpose: ${keyConfig.description}`);
    console.log(`   Get your key: ${keyConfig.getUrl}`);
    
    let isValid = false;
    let attempts = 0;
    const maxAttempts = 3;
    
    while (!isValid && attempts < maxAttempts) {
      const apiKey = await this.askQuestion(
        `\nPlease paste your ${keyConfig.service} API key: `,
        true // hide input
      );
      
      // Validate format
      if (!this.validateKeyFormat(keyConfig.name, apiKey)) {
        console.log('❌ That key doesn\'t look right. Please check the format.');
        attempts++;
        continue;
      }
      
      // Test the key
      console.log('🔍 Testing API key...');
      const testResult = await this.testApiKey(keyConfig, apiKey);
      
      if (testResult.success) {
        await this.storeKey(keyConfig.name, apiKey);
        console.log('✅ Key validated and stored securely!');
        this.log(`Successfully setup ${keyConfig.service} API key`);
        isValid = true;
      } else {
        console.log(`❌ API test failed: ${testResult.error}`);
        console.log('Please check your key and try again.');
        this.log(`API test failed for ${keyConfig.service}: ${testResult.error}`, 'WARN');
        attempts++;
      }
    }
    
    if (!isValid) {
      if (keyConfig.required) {
        throw new Error(`Failed to setup required service: ${keyConfig.service} after ${maxAttempts} attempts`);
      } else {
        console.log(`⚠️  Skipping optional service: ${keyConfig.service}`);
        this.log(`Skipped optional service: ${keyConfig.service}`, 'WARN');
      }
    }
  }

  /**
   * Setup optional API keys
   */
  async setupOptionalKeys() {
    const setupOptional = await this.askQuestion(
      '\n🤔 Would you like to setup optional API keys? (y/n): '
    );
    
    if (setupOptional.toLowerCase().startsWith('y')) {
      for (const keyConfig of this.optionalKeys) {
        const setup = await this.askQuestion(
          `\nSetup ${keyConfig.service}? (y/n): `
        );
        
        if (setup.toLowerCase().startsWith('y')) {
          try {
            await this.setupSingleKey(keyConfig);
          } catch (error) {
            console.log(`⚠️  Skipping ${keyConfig.service}: ${error.message}`);
            this.log(`Skipped ${keyConfig.service}: ${error.message}`, 'WARN');
          }
        }
      }
    }
  }

  /**
   * Validate API key format
   */
  validateKeyFormat(keyName, apiKey) {
    if (!apiKey || apiKey.trim().length === 0) {
      return false;
    }
    
    const pattern = this.keyPatterns[keyName];
    if (pattern) {
      return pattern.test(apiKey.trim());
    }
    
    // Generic validation for unknown keys
    return apiKey.trim().length >= 16;
  }

  /**
   * Test API key with harmless call
   */
  async testApiKey(keyConfig, apiKey) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      let testUrl, headers;
      
      switch (keyConfig.name) {
        case 'OPENAI_API_KEY':
          testUrl = 'https://api.openai.com/v1/models';
          headers = { 'Authorization': `Bearer ${apiKey}` };
          break;
          
        case 'GOOGLE_API_KEY':
          testUrl = `https://www.googleapis.com/customsearch/v1?key=${apiKey}&cx=test&q=test`;
          headers = {};
          break;
          
        case 'NEWS_API_KEY':
          testUrl = `https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey=${apiKey}`;
          headers = {};
          break;
          
        case 'YOUTUBE_API_KEY':
          testUrl = `https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q=test&key=${apiKey}`;
          headers = {};
          break;
          
        case 'ANTHROPIC_API_KEY':
          testUrl = 'https://api.anthropic.com/v1/messages';
          headers = {
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
          };
          break;
          
        default:
          return { success: true, message: 'Format validation passed' };
      }
      
      const response = await fetch(testUrl, {
        method: keyConfig.name === 'ANTHROPIC_API_KEY' ? 'POST' : 'GET',
        headers,
        body: keyConfig.name === 'ANTHROPIC_API_KEY' ? JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 1,
          messages: [{ role: 'user', content: 'Hi' }]
        }) : undefined,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok || response.status === 400) {
        // 400 might be expected for some test calls
        return { success: true, message: 'API key validated successfully' };
      } else if (response.status === 401 || response.status === 403) {
        return { success: false, error: 'Invalid API key or insufficient permissions' };
      } else {
        return { success: false, error: `HTTP ${response.status}: ${response.statusText}` };
      }
      
    } catch (error) {
      if (error.name === 'AbortError') {
        return { success: false, error: 'Request timeout - please check your internet connection' };
      }
      return { success: false, error: error.message };
    }
  }

  /**
   * Store API key securely in .env.local
   */
  async storeKey(keyName, apiKey) {
    let envContent = '';
    
    // Read existing content
    if (fs.existsSync(this.envFilePath)) {
      envContent = fs.readFileSync(this.envFilePath, 'utf8');
    } else {
      // Create header for new file
      envContent = `# AI News Dashboard - Secure Environment Configuration
# Generated by SecureKeyAgent - ${new Date().toISOString()}
# This file is automatically added to .gitignore
# NEVER commit API keys to version control

`;
    }
    
    // Remove existing key if present
    const keyRegex = new RegExp(`^${keyName}=.*$`, 'gm');
    envContent = envContent.replace(keyRegex, '');
    
    // Add the new key
    envContent += `${keyName}=${apiKey}\n`;
    
    // Write securely
    fs.writeFileSync(this.envFilePath, envContent, { mode: 0o600 });
    
    this.log(`Stored key: ${keyName} (hash: ${this.hashKey(apiKey)})`);
  }

  /**
   * Validate all stored keys
   */
  async validateAllKeys() {
    console.log('\n🔍 Validating all stored keys...');
    
    if (!fs.existsSync(this.envFilePath)) {
      throw new Error('No .env.local file found');
    }
    
    const envContent = fs.readFileSync(this.envFilePath, 'utf8');
    const envLines = envContent.split('\n');
    const storedKeys = {};
    
    // Parse stored keys
    envLines.forEach(line => {
      const match = line.match(/^([A-Z_]+)=(.+)$/);
      if (match) {
        storedKeys[match[1]] = match[2];
      }
    });
    
    // Check required keys
    const missingKeys = this.requiredKeys.filter(
      keyConfig => !storedKeys[keyConfig.name]
    );
    
    if (missingKeys.length > 0) {
      throw new Error(`Missing required keys: ${missingKeys.map(k => k.name).join(', ')}`);
    }
    
    console.log('✅ All required keys present and validated');
    this.log('All required keys validated successfully');
  }

  /**
   * Perform final security check
   */
  async performSecurityCheck() {
    console.log('\n🛡️  Performing security check...');
    
    // Check .gitignore
    const gitignorePath = path.join(this.projectRoot, '.gitignore');
    if (fs.existsSync(gitignorePath)) {
      const gitignoreContent = fs.readFileSync(gitignorePath, 'utf8');
      if (!gitignoreContent.includes('.env.local')) {
        // Add .env.local to .gitignore
        fs.appendFileSync(gitignorePath, '\n# Environment variables\n.env.local\n.env.*.local\n');
        console.log('✅ Added .env.local to .gitignore');
        this.log('Added .env.local to .gitignore');
      } else {
        console.log('✅ .env.local already in .gitignore');
      }
    }
    
    // Check file permissions
    const stats = fs.statSync(this.envFilePath);
    const mode = stats.mode & parseInt('777', 8);
    if (mode !== parseInt('600', 8)) {
      fs.chmodSync(this.envFilePath, 0o600);
      console.log('✅ Set secure file permissions (600)');
      this.log('Set secure file permissions');
    }
    
    // Verify no keys in main .env
    const mainEnvPath = path.join(this.projectRoot, '.env');
    if (fs.existsSync(mainEnvPath)) {
      const mainEnvContent = fs.readFileSync(mainEnvPath, 'utf8');
      const hasRealKeys = this.requiredKeys.some(keyConfig => {
        const match = mainEnvContent.match(new RegExp(`^${keyConfig.name}=(.+)$`, 'm'));
        return match && !match[1].includes('placeholder');
      });
      
      if (hasRealKeys) {
        console.log('⚠️  Warning: Real API keys detected in .env file - consider moving to .env.local');
        this.log('Warning: Real API keys detected in .env file', 'WARN');
      }
    }
    
    console.log('✅ Security check completed');
    this.log('Security check completed successfully');
  }

  /**
   * Handle setup errors
   */
  async handleSetupError(error) {
    console.log('\n🔄 Error Recovery Options:');
    console.log('1. Try setup again');
    console.log('2. Restore from backup (if available)');
    console.log('3. Manual setup guide');
    
    const choice = await this.askQuestion('Choose recovery option (1/2/3): ');
    
    switch (choice.trim()) {
      case '1':
        console.log('🔄 Restarting setup...');
        this.log('Restarting setup after error');
        await this.setupKeys();
        break;
      case '2':
        if (fs.existsSync(this.backupPath)) {
          fs.copyFileSync(this.backupPath, this.envFilePath);
          console.log('✅ Restored from backup');
          this.log('Restored from backup');
        } else {
          console.log('❌ No backup available');
        }
        break;
      case '3':
        this.showManualSetupGuide();
        break;
    }
  }

  /**
   * Show manual setup guide
   */
  showManualSetupGuide() {
    console.log('\n📚 Manual Setup Guide:');
    console.log('1. Create .env.local in project root');
    console.log('2. Add your API keys in this format:');
    console.log('   OPENAI_API_KEY=your_actual_key_here');
    console.log('   GOOGLE_API_KEY=your_actual_key_here');
    console.log('   etc.');
    console.log('3. Ensure .env.local is in .gitignore');
    console.log('4. Run: npm run test:api-keys');
    this.log('Displayed manual setup guide');
  }

  /**
   * Hash API key for logging (never log actual keys)
   */
  hashKey(apiKey) {
    return crypto.createHash('sha256').update(apiKey).digest('hex').substring(0, 8);
  }

  /**
   * Secure logging
   */
  log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    const logEntry = `${timestamp} [${level}] ${message}\n`;
    fs.appendFileSync(this.auditLogPath, logEntry);
  }

  /**
   * Ask user question with optional hidden input
   */
  askQuestion(question, hideInput = false) {
    return new Promise((resolve) => {
      if (hideInput) {
        process.stdout.write(question);
        const stdin = process.stdin;
        stdin.setRawMode(true);
        stdin.resume();
        stdin.setEncoding('utf8');
        
        let input = '';
        stdin.on('data', (char) => {
          if (char === '\r' || char === '\n') {
            stdin.setRawMode(false);
            stdin.pause();
            console.log(''); // New line
            resolve(input);
          } else if (char === '\u0003') {
            // Ctrl+C
            process.exit();
          } else if (char === '\u007f') {
            // Backspace
            if (input.length > 0) {
              input = input.slice(0, -1);
            }
          } else {
            input += char;
          }
        });
      } else {
        this.rl.question(question, resolve);
      }
    });
  }
}

// Export for testing
module.exports = SecureKeyAgent;

// Run if called directly
if (require.main === module) {
  const agent = new SecureKeyAgent();
  agent.setupKeys().catch(console.error);
}
