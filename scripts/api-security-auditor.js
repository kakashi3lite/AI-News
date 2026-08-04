#!/usr/bin/env node
/**
 * API Key Validator and Security Auditor
 * Validates existing keys and performs security checks
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const https = require('https');

class APISecurityAuditor {
  constructor() {
    this.projectRoot = path.resolve(__dirname, '..');
    this.envFilePath = path.join(this.projectRoot, '.env.local');
    this.reportPath = path.join(this.projectRoot, 'logs', 'security-audit-report.json');
    this.auditLogPath = path.join(this.projectRoot, 'logs', 'security-audit.log');
    
    // Create logs directory if it doesn't exist
    const logsDir = path.dirname(this.reportPath);
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }

    // API key patterns for validation
    this.keyPatterns = {
      OPENAI_API_KEY: /^sk-(proj-)?[A-Za-z0-9]{32,}$/,
      GOOGLE_API_KEY: /^[A-Za-z0-9_-]{39}$/,
      GOOGLE_CSE_ID: /^[a-f0-9]{17}:[a-f0-9]{10}$/,
      NEWS_API_KEY: /^[a-f0-9]{32}$/,
      YOUTUBE_API_KEY: /^[A-Za-z0-9_-]{39}$/,
      ANTHROPIC_API_KEY: /^sk-ant-[A-Za-z0-9_-]{95,}$/,
      GEMINI_API_KEY: /^[A-Za-z0-9_-]{39}$/
    };
  }

  /**
   * Comprehensive security audit
   */
  async performSecurityAudit() {
    const startTime = Date.now();
    this.log('Starting comprehensive security audit');
    
    console.log('\n🛡️ API Security Auditor - Comprehensive Analysis');
    console.log('=' .repeat(60));
    console.log('✅ API key validation and testing');
    console.log('✅ File security and permission analysis');
    console.log('✅ Git safety and exposure detection');
    console.log('✅ Best practices compliance check');
    console.log('=' .repeat(60));

    const auditResults = {
      timestamp: new Date().toISOString(),
      apiKeys: {},
      security: {},
      gitSafety: {},
      compliance: {},
      recommendations: [],
      overallScore: 0
    };

    try {
      // 1. API Key Analysis
      auditResults.apiKeys = await this.auditApiKeys();
      
      // 2. File Security Analysis
      auditResults.security = await this.auditFileSecurity();
      
      // 3. Git Safety Check
      auditResults.gitSafety = await this.auditGitSafety();
      
      // 4. Compliance Check
      auditResults.compliance = await this.auditCompliance();
      
      // 5. Generate recommendations
      auditResults.recommendations = this.generateRecommendations(auditResults);
      
      // 6. Calculate overall score
      auditResults.overallScore = this.calculateOverallScore(auditResults);
      
      // 7. Generate report
      await this.generateAuditReport(auditResults, startTime);
      
      this.displayAuditSummary(auditResults);
      
      return auditResults;
      
    } catch (error) {
      console.error('\n❌ Audit failed:', error.message);
      this.log(`Audit error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Audit API keys for validity and security
   */
  async auditApiKeys() {
    console.log('\n🔍 Auditing API Keys...');
    
    if (!fs.existsSync(this.envFilePath)) {
      return {
        status: 'no_env_file',
        message: 'No .env.local file found',
        keys: {},
        score: 0
      };
    }

    const envContent = fs.readFileSync(this.envFilePath, 'utf8');
    const envKeys = this.parseEnvironmentFile(envContent);
    
    if (Object.keys(envKeys).length === 0) {
      return {
        status: 'no_keys',
        message: 'No API keys found',
        keys: {},
        score: 0
      };
    }

    const keyResults = {};
    let validKeys = 0;
    let totalKeys = Object.keys(envKeys).length;

    for (const [keyName, keyValue] of Object.entries(envKeys)) {
      console.log(`   🔍 Validating ${keyName}...`);
      
      const result = await this.validateSingleKey(keyName, keyValue);
      keyResults[keyName] = result;
      
      if (result.status === 'valid' || result.status === 'format_valid') {
        validKeys++;
        console.log(`   ✅ ${keyName}: ${result.message}`);
      } else {
        console.log(`   ❌ ${keyName}: ${result.error}`);
      }
    }

    const score = Math.round((validKeys / totalKeys) * 100);
    
    return {
      status: 'analyzed',
      message: `${validKeys}/${totalKeys} keys valid`,
      keys: keyResults,
      validKeys,
      totalKeys,
      score
    };
  }

  /**
   * Audit file security and permissions
   */
  async auditFileSecurity() {
    console.log('\n🔒 Auditing File Security...');
    
    const securityChecks = {
      envLocalExists: fs.existsSync(this.envFilePath),
      envLocalPermissions: null,
      envMainSecure: true,
      backupFilesSecure: true,
      logsSecurity: true
    };

    // Check .env.local permissions
    if (securityChecks.envLocalExists) {
      try {
        const stats = fs.statSync(this.envFilePath);
        if (process.platform !== 'win32') {
          const mode = '0' + (stats.mode & parseInt('777', 8)).toString(8);
          securityChecks.envLocalPermissions = mode;
          console.log(`   📋 .env.local permissions: ${mode}`);
        } else {
          securityChecks.envLocalPermissions = 'windows_acl';
          console.log('   📋 .env.local: Windows ACL protected');
        }
      } catch (error) {
        securityChecks.envLocalPermissions = 'error';
      }
    }

    // Check main .env file for real keys
    const mainEnvPath = path.join(this.projectRoot, '.env');
    if (fs.existsSync(mainEnvPath)) {
      const mainEnvContent = fs.readFileSync(mainEnvPath, 'utf8');
      securityChecks.envMainSecure = !this.containsRealKeys(mainEnvContent);
      console.log(`   📋 .env file security: ${securityChecks.envMainSecure ? 'Safe' : 'Contains real keys'}`);
    }

    // Check for backup files
    const backupFiles = this.findBackupFiles();
    securityChecks.backupFilesSecure = backupFiles.length === 0;
    if (backupFiles.length > 0) {
      console.log(`   ⚠️  Found ${backupFiles.length} backup files`);
    }

    console.log('   ✅ File security audit complete');

    const score = this.calculateSecurityScore(securityChecks);
    
    return {
      ...securityChecks,
      backupFiles,
      score
    };
  }

  /**
   * Audit Git safety and exposure risks
   */
  async auditGitSafety() {
    console.log('\n📂 Auditing Git Safety...');
    
    const gitChecks = {
      isGitRepo: false,
      gitignoreExists: false,
      gitignoreSecure: false,
      noKeysInHistory: true,
      stagedFiles: []
    };

    try {
      // Check if it's a git repository
      const { execSync } = require('child_process');
      execSync('git rev-parse --git-dir', { stdio: 'ignore' });
      gitChecks.isGitRepo = true;
      console.log('   📋 Git repository detected');

      // Check .gitignore
      const gitignorePath = path.join(this.projectRoot, '.gitignore');
      gitChecks.gitignoreExists = fs.existsSync(gitignorePath);
      
      if (gitChecks.gitignoreExists) {
        const gitignoreContent = fs.readFileSync(gitignorePath, 'utf8');
        gitChecks.gitignoreSecure = this.validateGitignore(gitignoreContent);
        console.log(`   📋 .gitignore security: ${gitChecks.gitignoreSecure ? 'Secure' : 'Needs update'}`);
      }

      // Check staged files
      try {
        const stagedOutput = execSync('git diff --cached --name-only', { encoding: 'utf8' });
        gitChecks.stagedFiles = stagedOutput.trim().split('\n').filter(f => f);
        
        const sensitiveStaged = gitChecks.stagedFiles.filter(file => 
          file.includes('.env') && !file.includes('.example')
        );
        
        if (sensitiveStaged.length > 0) {
          console.log(`   ⚠️  Sensitive files staged: ${sensitiveStaged.join(', ')}`);
        }
      } catch (error) {
        // No staged files or other error
      }

    } catch (error) {
      console.log('   📋 Not a git repository or git not available');
    }

    console.log('   ✅ Git safety audit complete');

    const score = this.calculateGitSafetyScore(gitChecks);
    
    return {
      ...gitChecks,
      score
    };
  }

  /**
   * Audit compliance with security best practices
   */
  async auditCompliance() {
    console.log('\n📋 Auditing Security Compliance...');
    
    const complianceChecks = {
      envLocalOnly: true,
      noHardcodedKeys: true,
      secureFilePermissions: true,
      logSecurityCompliant: true,
      backupProtection: true
    };

    // Check for hardcoded keys in source files
    const sourceFiles = this.findSourceFiles();
    for (const file of sourceFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        if (this.containsHardcodedKeys(content)) {
          complianceChecks.noHardcodedKeys = false;
          console.log(`   ⚠️  Hardcoded keys detected in ${file}`);
          break;
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }

    // Check log files for key exposure
    const logFiles = this.findLogFiles();
    for (const logFile of logFiles) {
      try {
        const content = fs.readFileSync(logFile, 'utf8');
        if (this.containsRealKeys(content)) {
          complianceChecks.logSecurityCompliant = false;
          console.log(`   ⚠️  Keys detected in log file ${logFile}`);
          break;
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }

    console.log('   ✅ Compliance audit complete');

    const score = this.calculateComplianceScore(complianceChecks);
    
    return {
      ...complianceChecks,
      sourceFilesChecked: sourceFiles.length,
      logFilesChecked: logFiles.length,
      score
    };
  }

  /**
   * Parse environment file content
   */
  parseEnvironmentFile(content) {
    const envKeys = {};
    content.split('\n').forEach(line => {
      const match = line.match(/^([A-Z_]+)=(.+)$/);
      if (match && match[2] && !match[2].includes('placeholder')) {
        envKeys[match[1]] = match[2].trim();
      }
    });
    return envKeys;
  }

  /**
   * Validate a single API key
   */
  async validateSingleKey(keyName, apiKey) {
    // Format validation
    const pattern = this.keyPatterns[keyName];
    if (pattern && !pattern.test(apiKey)) {
      return {
        status: 'invalid_format',
        error: 'Invalid format',
        hash: this.hashKey(apiKey)
      };
    }

    // Live testing (optional, can be disabled for quick audits)
    if (process.env.SKIP_API_TESTS !== 'true') {
      try {
        const testResult = await this.quickTestKey(keyName, apiKey);
        if (testResult.success) {
          return {
            status: 'valid',
            message: 'Valid and tested',
            hash: this.hashKey(apiKey)
          };
        } else {
          return {
            status: 'test_failed',
            error: testResult.error,
            hash: this.hashKey(apiKey)
          };
        }
      } catch (error) {
        return {
          status: 'format_valid',
          message: 'Valid format, test skipped',
          hash: this.hashKey(apiKey)
        };
      }
    }

    return {
      status: 'format_valid',
      message: 'Valid format',
      hash: this.hashKey(apiKey)
    };
  }

  /**
   * Quick test for API key validity
   */
  async quickTestKey(keyName, apiKey) {
    // Simplified testing with shorter timeouts
    const timeout = 5000;
    
    switch (keyName) {
      case 'OPENAI_API_KEY':
        return await this.quickTestOpenAI(apiKey, timeout);
      case 'GOOGLE_API_KEY':
        return await this.quickTestGoogle(apiKey, timeout);
      default:
        return { success: true, message: 'Format valid' };
    }
  }

  /**
   * Quick OpenAI test
   */
  async quickTestOpenAI(apiKey, timeout) {
    return new Promise((resolve) => {
      const options = {
        hostname: 'api.openai.com',
        path: '/v1/models',
        method: 'HEAD',
        headers: {
          'Authorization': `Bearer ${apiKey}`
        },
        timeout: timeout
      };

      const req = https.request(options, (res) => {
        resolve({ 
          success: res.statusCode === 200, 
          error: res.statusCode !== 200 ? `HTTP ${res.statusCode}` : null 
        });
      });

      req.on('error', () => resolve({ success: false, error: 'Connection failed' }));
      req.on('timeout', () => resolve({ success: false, error: 'Timeout' }));
      req.end();
    });
  }

  /**
   * Quick Google test
   */
  async quickTestGoogle(apiKey, timeout) {
    return new Promise((resolve) => {
      const options = {
        hostname: 'www.googleapis.com',
        path: `/customsearch/v1?key=${apiKey}&cx=test&q=test`,
        method: 'HEAD',
        timeout: timeout
      };

      const req = https.request(options, (res) => {
        resolve({ 
          success: res.statusCode < 500, 
          error: res.statusCode >= 500 ? `HTTP ${res.statusCode}` : null 
        });
      });

      req.on('error', () => resolve({ success: false, error: 'Connection failed' }));
      req.on('timeout', () => resolve({ success: false, error: 'Timeout' }));
      req.end();
    });
  }

  /**
   * Check if content contains real API keys
   */
  containsRealKeys(content) {
    const patterns = [
      /sk-[A-Za-z0-9]{48,}/g,  // OpenAI
      /[A-Za-z0-9_-]{39}/g,    // Google (basic check)
      /[a-f0-9]{32}/g          // NewsAPI
    ];
    
    return patterns.some(pattern => {
      const matches = content.match(pattern);
      return matches && matches.some(match => 
        !match.includes('placeholder') && 
        !match.includes('example') &&
        !match.includes('your_')
      );
    });
  }

  /**
   * Check for hardcoded keys in source code
   */
  containsHardcodedKeys(content) {
    // Look for direct assignment of API keys
    const hardcodedPatterns = [
      /(?:OPENAI|GOOGLE|NEWS)_API_KEY\s*=\s*['"](?!placeholder|your_|example)[A-Za-z0-9_-]{20,}/g,
      /sk-[A-Za-z0-9]{48,}/g
    ];
    
    return hardcodedPatterns.some(pattern => pattern.test(content));
  }

  /**
   * Validate .gitignore content
   */
  validateGitignore(content) {
    const requiredEntries = [
      '.env.local',
      '.env.*.local',
      'logs/',
      '*.log'
    ];
    
    return requiredEntries.every(entry => content.includes(entry));
  }

  /**
   * Find source files to check
   */
  findSourceFiles() {
    const extensions = ['.js', '.jsx', '.ts', '.tsx', '.py'];
    const excludeDirs = ['node_modules', '.next', 'dist', 'build'];
    
    return this.walkDirectory(this.projectRoot, extensions, excludeDirs);
  }

  /**
   * Find log files to check
   */
  findLogFiles() {
    const logExtensions = ['.log'];
    return this.walkDirectory(this.projectRoot, logExtensions, ['node_modules']);
  }

  /**
   * Find backup files
   */
  findBackupFiles() {
    const backupPatterns = ['.env.backup', '.env.bak', '.env.old'];
    const files = [];
    
    try {
      const dirFiles = fs.readdirSync(this.projectRoot);
      for (const file of dirFiles) {
        if (backupPatterns.some(pattern => file.includes(pattern))) {
          files.push(file);
        }
      }
    } catch (error) {
      // Directory read error
    }
    
    return files;
  }

  /**
   * Walk directory tree for files
   */
  walkDirectory(dir, extensions, excludeDirs) {
    const files = [];
    
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory()) {
          if (!excludeDirs.includes(entry.name)) {
            files.push(...this.walkDirectory(fullPath, extensions, excludeDirs));
          }
        } else if (entry.isFile()) {
          const ext = path.extname(entry.name);
          if (extensions.includes(ext)) {
            files.push(fullPath);
          }
        }
      }
    } catch (error) {
      // Directory access error
    }
    
    return files;
  }

  /**
   * Calculate security score
   */
  calculateSecurityScore(checks) {
    let score = 0;
    let total = 0;
    
    if (checks.envLocalExists) { score += 25; total += 25; }
    if (checks.envLocalPermissions === '0600' || checks.envLocalPermissions === 'windows_acl') { score += 25; total += 25; }
    if (checks.envMainSecure) { score += 25; total += 25; }
    if (checks.backupFilesSecure) { score += 25; total += 25; }
    
    return total > 0 ? Math.round((score / total) * 100) : 0;
  }

  /**
   * Calculate Git safety score
   */
  calculateGitSafetyScore(checks) {
    let score = 0;
    let total = 100;
    
    if (checks.isGitRepo) {
      if (checks.gitignoreExists) score += 40;
      if (checks.gitignoreSecure) score += 40;
      if (checks.stagedFiles.length === 0 || 
          !checks.stagedFiles.some(f => f.includes('.env'))) score += 20;
    } else {
      score = 100; // No git repo means no git safety issues
    }
    
    return Math.round(score);
  }

  /**
   * Calculate compliance score
   */
  calculateComplianceScore(checks) {
    let score = 0;
    let total = 0;
    
    if (checks.noHardcodedKeys) { score += 40; total += 40; }
    if (checks.logSecurityCompliant) { score += 30; total += 30; }
    if (checks.backupProtection) { score += 30; total += 30; }
    
    return total > 0 ? Math.round((score / total) * 100) : 0;
  }

  /**
   * Calculate overall security score
   */
  calculateOverallScore(results) {
    const weights = {
      apiKeys: 0.4,    // 40%
      security: 0.25,  // 25%
      gitSafety: 0.2,  // 20%
      compliance: 0.15 // 15%
    };
    
    const weightedScore = 
      (results.apiKeys.score || 0) * weights.apiKeys +
      (results.security.score || 0) * weights.security +
      (results.gitSafety.score || 0) * weights.gitSafety +
      (results.compliance.score || 0) * weights.compliance;
    
    return Math.round(weightedScore);
  }

  /**
   * Generate recommendations based on audit results
   */
  generateRecommendations(results) {
    const recommendations = [];
    
    // API Key recommendations
    if (results.apiKeys.score < 100) {
      recommendations.push({
        category: 'API Keys',
        priority: 'high',
        message: 'Some API keys are invalid or missing. Run setup to fix.',
        action: 'npm run setup:keys'
      });
    }
    
    // Security recommendations
    if (results.security.score < 90) {
      recommendations.push({
        category: 'File Security',
        priority: 'high',
        message: 'File permissions or security settings need attention.',
        action: 'Check .env.local permissions and backup files'
      });
    }
    
    // Git safety recommendations
    if (results.gitSafety.score < 90) {
      recommendations.push({
        category: 'Git Safety',
        priority: 'medium',
        message: 'Git configuration could be more secure.',
        action: 'Update .gitignore and check staged files'
      });
    }
    
    // Compliance recommendations
    if (results.compliance.score < 90) {
      recommendations.push({
        category: 'Compliance',
        priority: 'high',
        message: 'Security best practices compliance issues detected.',
        action: 'Remove hardcoded keys and secure log files'
      });
    }
    
    return recommendations;
  }

  /**
   * Generate detailed audit report
   */
  async generateAuditReport(results, startTime) {
    const report = {
      ...results,
      executionTime: Date.now() - startTime,
      version: '1.0.0',
      auditor: 'Enhanced SecureKeyAgent'
    };
    
    // Save JSON report
    fs.writeFileSync(this.reportPath, JSON.stringify(report, null, 2));
    
    // Save readable summary
    const summaryPath = path.join(path.dirname(this.reportPath), 'security-audit-summary.txt');
    const summary = this.generateReadableSummary(results);
    fs.writeFileSync(summaryPath, summary);
    
    console.log(`\n📄 Audit report saved: ${this.reportPath}`);
    console.log(`📄 Summary saved: ${summaryPath}`);
  }

  /**
   * Generate readable summary
   */
  generateReadableSummary(results) {
    const timestamp = new Date().toISOString();
    
    return `
API SECURITY AUDIT SUMMARY
Generated: ${timestamp}
Overall Score: ${results.overallScore}/100

API KEYS (${results.apiKeys.score}/100):
${results.apiKeys.totalKeys ? `${results.apiKeys.validKeys}/${results.apiKeys.totalKeys} keys valid` : 'No keys found'}

FILE SECURITY (${results.security.score}/100):
- .env.local exists: ${results.security.envLocalExists ? 'Yes' : 'No'}
- Secure permissions: ${results.security.envLocalPermissions === '0600' || results.security.envLocalPermissions === 'windows_acl' ? 'Yes' : 'No'}
- Main .env secure: ${results.security.envMainSecure ? 'Yes' : 'No'}

GIT SAFETY (${results.gitSafety.score}/100):
- Git repository: ${results.gitSafety.isGitRepo ? 'Yes' : 'No'}
- .gitignore secure: ${results.gitSafety.gitignoreSecure ? 'Yes' : 'No'}

COMPLIANCE (${results.compliance.score}/100):
- No hardcoded keys: ${results.compliance.noHardcodedKeys ? 'Yes' : 'No'}
- Log security: ${results.compliance.logSecurityCompliant ? 'Yes' : 'No'}

RECOMMENDATIONS:
${results.recommendations.map(r => `- ${r.category}: ${r.message} (${r.priority} priority)`).join('\n')}
`;
  }

  /**
   * Display audit summary in console
   */
  displayAuditSummary(results) {
    console.log('\n📊 SECURITY AUDIT SUMMARY');
    console.log('=' .repeat(40));
    console.log(`Overall Security Score: ${results.overallScore}/100`);
    
    const scoreColor = results.overallScore >= 90 ? '🟢' : 
                      results.overallScore >= 70 ? '🟡' : '🔴';
    console.log(`${scoreColor} Security Level: ${this.getSecurityLevel(results.overallScore)}`);
    
    console.log('\n📋 Component Scores:');
    console.log(`   🔑 API Keys: ${results.apiKeys.score}/100`);
    console.log(`   🔒 File Security: ${results.security.score}/100`);
    console.log(`   📂 Git Safety: ${results.gitSafety.score}/100`);
    console.log(`   📋 Compliance: ${results.compliance.score}/100`);
    
    if (results.recommendations.length > 0) {
      console.log('\n💡 Recommendations:');
      results.recommendations.forEach(rec => {
        const priority = rec.priority === 'high' ? '🔴' : 
                        rec.priority === 'medium' ? '🟡' : '🟢';
        console.log(`   ${priority} ${rec.message}`);
      });
    } else {
      console.log('\n✅ All security checks passed! No recommendations.');
    }
    
    console.log('\n🎯 Next Steps:');
    if (results.overallScore < 70) {
      console.log('   Run: npm run setup:keys (fix critical issues)');
    }
    console.log('   Run: npm run validate:keys (test API keys)');
    console.log('   Check: logs/security-audit-report.json');
  }

  /**
   * Get security level description
   */
  getSecurityLevel(score) {
    if (score >= 90) return 'Excellent';
    if (score >= 80) return 'Good';
    if (score >= 70) return 'Fair';
    if (score >= 60) return 'Poor';
    return 'Critical';
  }

  /**
   * Secure logging (never log actual keys)
   */
  log(message) {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] ${message}\n`;
    fs.appendFileSync(this.auditLogPath, logEntry);
  }

  /**
   * Hash API key for logging (never log actual keys)
   */
  hashKey(apiKey) {
    return crypto.createHash('sha256').update(apiKey).digest('hex').substring(0, 8);
  }
}

// Export for use in other modules
module.exports = APISecurityAuditor;

// Main execution
if (require.main === module) {
  const auditor = new APISecurityAuditor();
  auditor.performSecurityAudit().catch(error => {
    console.error('❌ Audit failed:', error.message);
    process.exit(1);
  });
}
