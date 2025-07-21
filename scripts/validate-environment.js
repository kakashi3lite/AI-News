#!/usr/bin/env node

/**
 * SecureKeyAgent - Environment Validator
 * Comprehensive environment validation for production deployment
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class EnvironmentValidator {
  constructor() {
    this.projectRoot = path.resolve(__dirname, '..');
    this.envFilePath = path.join(this.projectRoot, '.env.local');
    this.envExamplePath = path.join(this.projectRoot, '.env.example');
    this.gitignorePath = path.join(this.projectRoot, '.gitignore');
    this.reportPath = path.join(this.projectRoot, 'logs', 'environment-validation.json');
    
    // Create logs directory if it doesn't exist
    const logsDir = path.dirname(this.reportPath);
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    
    // Security patterns to check for
    this.sensitivePatterns = [
      /(?:api[_-]?key|secret|token|password)\s*[:=]\s*["']?[a-zA-Z0-9_-]{8,}["']?/gi,
      /sk-[a-zA-Z0-9]{32,}/g, // OpenAI keys
      /xoxb-[a-zA-Z0-9-]+/g,  // Slack tokens
      /ghp_[a-zA-Z0-9]{36}/g,  // GitHub tokens
      /AKIA[0-9A-Z]{16}/g      // AWS access keys
    ];
    
    // Required environment variables
    this.requiredVars = {
      NODE_ENV: {
        required: true,
        values: ['development', 'production', 'test'],
        description: 'Environment mode'
      },
      NEXTAUTH_SECRET: {
        required: true,
        minLength: 32,
        description: 'NextAuth.js secret key'
      },
      NEXTAUTH_URL: {
        required: true,
        pattern: /^https?:\/\/.+/,
        description: 'NextAuth.js callback URL'
      }
    };
    
    // API keys configuration
    this.apiKeys = {
      OPENAI_API_KEY: {
        required: false,
        pattern: /^sk-(proj-)?[A-Za-z0-9]{32,}$/,
        description: 'OpenAI API key for AI features'
      },
      GOOGLE_API_KEY: {
        required: false,
        pattern: /^[A-Za-z0-9_-]{39}$/,
        description: 'Google API key for search'
      },
      NEWS_API_KEY: {
        required: false,
        pattern: /^[a-f0-9]{32}$/,
        description: 'NewsAPI key for news fetching'
      }
    };
  }

  /**
   * Main validation workflow
   */
  async validateEnvironment() {
    console.log('\n🔍 SecureKeyAgent - Environment Validator');
    console.log('=' .repeat(50));
    console.log('✅ Security validation');
    console.log('✅ Environment configuration check');
    console.log('✅ File permissions audit');
    console.log('✅ Production readiness assessment');
    console.log('=' .repeat(50));
    
    const results = {
      timestamp: new Date().toISOString(),
      overall: 'unknown',
      checks: {},
      security: {},
      recommendations: []
    };
    
    try {
      // 1. Environment file validation
      console.log('\n📁 Checking environment files...');
      results.checks.environmentFiles = await this.checkEnvironmentFiles();
      
      // 2. Environment variables validation
      console.log('\n🔧 Validating environment variables...');
      results.checks.environmentVars = await this.checkEnvironmentVariables();
      
      // 3. Security audit
      console.log('\n🛡️  Running security audit...');
      results.security = await this.runSecurityAudit();
      
      // 4. File permissions check
      console.log('\n🔒 Checking file permissions...');
      results.checks.permissions = await this.checkFilePermissions();
      
      // 5. Git configuration check
      console.log('\n📦 Checking Git configuration...');
      results.checks.gitConfig = await this.checkGitConfiguration();
      
      // 6. Production readiness
      console.log('\n🚀 Assessing production readiness...');
      results.checks.productionReadiness = await this.checkProductionReadiness();
      
      // Calculate overall status
      results.overall = this.calculateOverallStatus(results);
      
      // Generate recommendations
      results.recommendations = this.generateRecommendations(results);
      
      // Display results
      this.displayResults(results);
      
      // Save report
      fs.writeFileSync(this.reportPath, JSON.stringify(results, null, 2));
      
      return results;
      
    } catch (error) {
      console.error('\n❌ Environment validation failed:', error.message);
      results.overall = 'failed';
      results.error = error.message;
      return results;
    }
  }

  /**
   * Check environment files
   */
  async checkEnvironmentFiles() {
    const checks = {};
    
    // Check .env.local
    checks.envLocal = {
      exists: fs.existsSync(this.envFilePath),
      secure: false,
      size: 0
    };
    
    if (checks.envLocal.exists) {
      const stats = fs.statSync(this.envFilePath);
      checks.envLocal.size = stats.size;
      checks.envLocal.secure = (stats.mode & parseInt('077', 8)) === 0;
      console.log('   ✅ .env.local exists');
    } else {
      console.log('   ⚠️  .env.local not found');
    }
    
    // Check .env.example
    checks.envExample = {
      exists: fs.existsSync(this.envExamplePath)
    };
    
    if (checks.envExample.exists) {
      console.log('   ✅ .env.example exists');
    } else {
      console.log('   ⚠️  .env.example not found');
    }
    
    return checks;
  }

  /**
   * Check environment variables
   */
  async checkEnvironmentVariables() {
    const checks = {
      required: {},
      apiKeys: {},
      missing: [],
      invalid: []
    };
    
    // Load environment variables
    if (fs.existsSync(this.envFilePath)) {
      require('dotenv').config({ path: this.envFilePath });
    }
    
    // Check required variables
    for (const [varName, config] of Object.entries(this.requiredVars)) {
      const value = process.env[varName];
      const check = {
        present: !!value,
        valid: false,
        error: null
      };
      
      if (value) {
        if (config.values && !config.values.includes(value)) {
          check.error = `Invalid value. Expected: ${config.values.join(', ')}`;
        } else if (config.minLength && value.length < config.minLength) {
          check.error = `Too short. Minimum length: ${config.minLength}`;
        } else if (config.pattern && !config.pattern.test(value)) {
          check.error = 'Invalid format';
        } else {
          check.valid = true;
        }
      } else if (config.required) {
        checks.missing.push(varName);
      }
      
      checks.required[varName] = check;
      
      if (check.present && check.valid) {
        console.log(`   ✅ ${varName}: Valid`);
      } else if (check.present && !check.valid) {
        console.log(`   ❌ ${varName}: ${check.error}`);
        checks.invalid.push(varName);
      } else if (config.required) {
        console.log(`   ❌ ${varName}: Missing (required)`);
      } else {
        console.log(`   ⚠️  ${varName}: Missing (optional)`);
      }
    }
    
    // Check API keys
    for (const [keyName, config] of Object.entries(this.apiKeys)) {
      const value = process.env[keyName];
      const check = {
        present: !!value,
        valid: false,
        format: false
      };
      
      if (value && !value.includes('placeholder')) {
        check.format = config.pattern ? config.pattern.test(value) : true;
        check.valid = check.format;
      }
      
      checks.apiKeys[keyName] = check;
      
      if (check.present && check.valid) {
        console.log(`   ✅ ${keyName}: Present and valid format`);
      } else if (check.present && !check.valid) {
        console.log(`   ❌ ${keyName}: Present but invalid format`);
        checks.invalid.push(keyName);
      } else {
        console.log(`   ⚠️  ${keyName}: Not configured`);
      }
    }
    
    return checks;
  }

  /**
   * Run security audit
   */
  async runSecurityAudit() {
    const audit = {
      exposedSecrets: [],
      filePermissions: {},
      gitignoreCheck: {},
      sensitiveFileCheck: {}
    };
    
    // Check for exposed secrets in source files
    const sourceFiles = this.findSourceFiles();
    for (const file of sourceFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        for (const pattern of this.sensitivePatterns) {
          const matches = content.match(pattern);
          if (matches) {
            audit.exposedSecrets.push({
              file: path.relative(this.projectRoot, file),
              matches: matches.length,
              pattern: pattern.source
            });
          }
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    // Check .gitignore
    if (fs.existsSync(this.gitignorePath)) {
      const gitignoreContent = fs.readFileSync(this.gitignorePath, 'utf8');
      audit.gitignoreCheck = {
        exists: true,
        hasEnvLocal: gitignoreContent.includes('.env.local'),
        hasEnvFiles: gitignoreContent.includes('.env'),
        hasNodeModules: gitignoreContent.includes('node_modules'),
        hasDist: gitignoreContent.includes('dist') || gitignoreContent.includes('.next')
      };
    } else {
      audit.gitignoreCheck = { exists: false };
    }
    
    // Security summary
    const secretsFound = audit.exposedSecrets.length;
    const gitignoreOk = audit.gitignoreCheck.hasEnvLocal && audit.gitignoreCheck.hasNodeModules;
    
    if (secretsFound === 0 && gitignoreOk) {
      console.log('   ✅ No security issues found');
    } else {
      if (secretsFound > 0) {
        console.log(`   ❌ Found ${secretsFound} potential secret exposure(s)`);
      }
      if (!gitignoreOk) {
        console.log('   ❌ .gitignore configuration needs improvement');
      }
    }
    
    return audit;
  }

  /**
   * Check file permissions
   */
  async checkFilePermissions() {
    const checks = {};
    
    const filesToCheck = [
      { path: this.envFilePath, name: '.env.local', expectedMode: '600' },
      { path: this.gitignorePath, name: '.gitignore', expectedMode: '644' }
    ];
    
    for (const file of filesToCheck) {
      if (fs.existsSync(file.path)) {
        const stats = fs.statSync(file.path);
        const mode = (stats.mode & parseInt('777', 8)).toString(8);
        
        checks[file.name] = {
          exists: true,
          mode: mode,
          secure: mode === file.expectedMode || process.platform === 'win32' // Windows permissions work differently
        };
        
        if (checks[file.name].secure) {
          console.log(`   ✅ ${file.name}: Secure permissions (${mode})`);
        } else {
          console.log(`   ⚠️  ${file.name}: Permissions ${mode}, recommended ${file.expectedMode}`);
        }
      } else {
        checks[file.name] = { exists: false };
        console.log(`   ❌ ${file.name}: Not found`);
      }
    }
    
    return checks;
  }

  /**
   * Check Git configuration
   */
  async checkGitConfiguration() {
    const checks = {
      isRepo: false,
      hasGitignore: false,
      envFilesIgnored: false,
      hasCommits: false
    };
    
    try {
      // Check if it's a git repository
      const gitDir = path.join(this.projectRoot, '.git');
      checks.isRepo = fs.existsSync(gitDir);
      
      if (checks.isRepo) {
        console.log('   ✅ Git repository detected');
        
        // Check .gitignore
        checks.hasGitignore = fs.existsSync(this.gitignorePath);
        if (checks.hasGitignore) {
          const gitignoreContent = fs.readFileSync(this.gitignorePath, 'utf8');
          checks.envFilesIgnored = gitignoreContent.includes('.env.local');
          
          if (checks.envFilesIgnored) {
            console.log('   ✅ Environment files properly ignored');
          } else {
            console.log('   ❌ Environment files not ignored in .gitignore');
          }
        } else {
          console.log('   ❌ .gitignore file missing');
        }
      } else {
        console.log('   ⚠️  Not a Git repository');
      }
    } catch (error) {
      console.log('   ❌ Git configuration check failed');
    }
    
    return checks;
  }

  /**
   * Check production readiness
   */
  async checkProductionReadiness() {
    const checks = {
      environment: false,
      security: false,
      apiKeys: false,
      dependencies: false,
      score: 0
    };
    
    // Environment configuration
    checks.environment = process.env.NODE_ENV === 'production' || 
                        (process.env.NEXTAUTH_SECRET && process.env.NEXTAUTH_URL);
    
    // Security configuration
    checks.security = fs.existsSync(this.envFilePath) && 
                     fs.existsSync(this.gitignorePath);
    
    // API keys configuration
    const hasRequiredKeys = process.env.OPENAI_API_KEY || process.env.GOOGLE_API_KEY;
    checks.apiKeys = !!hasRequiredKeys;
    
    // Dependencies (check if package.json exists and has scripts)
    const packageJsonPath = path.join(this.projectRoot, 'package.json');
    if (fs.existsSync(packageJsonPath)) {
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
      checks.dependencies = !!(packageJson.scripts && packageJson.scripts.build);
    }
    
    // Calculate score
    const totalChecks = Object.keys(checks).length - 1; // Exclude score itself
    const passedChecks = Object.values(checks).filter(Boolean).length;
    checks.score = Math.round((passedChecks / totalChecks) * 100);
    
    console.log(`   📊 Production readiness score: ${checks.score}%`);
    
    if (checks.score >= 80) {
      console.log('   ✅ Ready for production deployment');
    } else if (checks.score >= 60) {
      console.log('   ⚠️  Mostly ready, some improvements needed');
    } else {
      console.log('   ❌ Not ready for production');
    }
    
    return checks;
  }

  /**
   * Find source files for security scanning
   */
  findSourceFiles() {
    const sourceFiles = [];
    const extensions = ['.js', '.ts', '.jsx', '.tsx', '.json', '.env'];
    const excludeDirs = ['node_modules', '.git', '.next', 'dist', 'build'];
    
    const scanDir = (dir) => {
      try {
        const items = fs.readdirSync(dir);
        for (const item of items) {
          const fullPath = path.join(dir, item);
          const stat = fs.statSync(fullPath);
          
          if (stat.isDirectory() && !excludeDirs.includes(item)) {
            scanDir(fullPath);
          } else if (stat.isFile()) {
            const ext = path.extname(item);
            if (extensions.includes(ext) || item.startsWith('.env')) {
              sourceFiles.push(fullPath);
            }
          }
        }
      } catch (error) {
        // Skip directories we can't read
      }
    };
    
    scanDir(this.projectRoot);
    return sourceFiles.slice(0, 100); // Limit to first 100 files for performance
  }

  /**
   * Calculate overall status
   */
  calculateOverallStatus(results) {
    const { checks, security } = results;
    
    // Critical issues
    if (security.exposedSecrets.length > 0) {
      return 'critical';
    }
    
    if (checks.environmentVars.missing.length > 0) {
      return 'error';
    }
    
    // Check production readiness
    if (checks.productionReadiness.score >= 80) {
      return 'ready';
    } else if (checks.productionReadiness.score >= 60) {
      return 'warning';
    } else {
      return 'not_ready';
    }
  }

  /**
   * Generate recommendations
   */
  generateRecommendations(results) {
    const recommendations = [];
    
    // Security recommendations
    if (results.security.exposedSecrets.length > 0) {
      recommendations.push({
        type: 'security',
        severity: 'critical',
        message: 'Potential secrets found in source code',
        action: 'Remove hardcoded secrets and use environment variables'
      });
    }
    
    // Environment recommendations
    if (results.checks.environmentVars.missing.length > 0) {
      recommendations.push({
        type: 'environment',
        severity: 'high',
        message: `Missing required environment variables: ${results.checks.environmentVars.missing.join(', ')}`,
        action: 'Run npm run setup:keys to configure missing variables'
      });
    }
    
    // Git recommendations
    if (!results.security.gitignoreCheck.hasEnvLocal) {
      recommendations.push({
        type: 'security',
        severity: 'high',
        message: 'Environment files not properly ignored in Git',
        action: 'Add .env.local to .gitignore'
      });
    }
    
    // Production readiness
    if (results.checks.productionReadiness.score < 80) {
      recommendations.push({
        type: 'deployment',
        severity: 'medium',
        message: 'Application not fully ready for production',
        action: 'Complete environment configuration and security setup'
      });
    }
    
    return recommendations;
  }

  /**
   * Display validation results
   */
  displayResults(results) {
    console.log('\n📊 Environment Validation Summary:');
    console.log('=' .repeat(40));
    
    const statusEmoji = {
      ready: '✅',
      warning: '⚠️ ',
      error: '❌',
      critical: '🔥',
      not_ready: '❌'
    };
    
    console.log(`${statusEmoji[results.overall]} Overall Status: ${results.overall.toUpperCase()}`);
    console.log(`📈 Production Readiness: ${results.checks.productionReadiness.score}%`);
    
    if (results.recommendations.length > 0) {
      console.log('\n💡 Recommendations:');
      results.recommendations.forEach((rec, index) => {
        const severityEmoji = { critical: '🔥', high: '❌', medium: '⚠️ ', low: 'ℹ️ ' };
        console.log(`   ${index + 1}. ${severityEmoji[rec.severity]} ${rec.message}`);
        console.log(`      Action: ${rec.action}`);
      });
    }
    
    console.log(`\n📄 Detailed report saved to: ${path.relative(this.projectRoot, this.reportPath)}`);
  }
}

// Export for testing
module.exports = EnvironmentValidator;

// Run if called directly
if (require.main === module) {
  const validator = new EnvironmentValidator();
  validator.validateEnvironment()
    .then(results => {
      const exitCode = ['ready', 'warning'].includes(results.overall) ? 0 : 1;
      process.exit(exitCode);
    })
    .catch(error => {
      console.error('Validation failed:', error);
      process.exit(1);
    });
}
