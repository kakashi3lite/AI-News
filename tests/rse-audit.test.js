/**
 * RSE Quality Auditor Test Suite
 * Comprehensive tests for the RSE Quality Auditor Bot v1
 * 
 * Test Categories:
 * - Unit tests for individual audit functions
 * - Integration tests for full audit pipeline
 * - Schema validation tests
 * - Markdown style validation tests
 * - Test coverage analysis
 * - Self-audit functionality
 */

const { RSEQualityAuditor } = require('../rse-audit-bot.js');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class RSEAuditTester {
  constructor() {
    this.testResults = {
      passed: 0,
      failed: 0,
      errors: [],
      coverage: 0
    };
    this.tempDir = path.join(__dirname, '../temp-audit-test');
    this.auditor = null;
  }

  async runAllTests() {
    console.log('🧪 Starting RSE Quality Auditor Test Suite\n');
    
    try {
      await this.setupTestEnvironment();
      
      // Unit tests
      await this.testAuditorInitialization();
      await this.testSchemaValidation();
      await this.testMarkdownValidation();
      await this.testTestExecution();
      await this.testLintingExecution();
      await this.testCoverageCalculation();
      await this.testSelfAudit();
      
      // Integration tests
      await this.testFullAuditPipeline();
      await this.testPREventHandling();
      await this.testReportGeneration();
      
      // Edge cases and error handling
      await this.testErrorHandling();
      await this.testMissingFiles();
      await this.testInvalidJSON();
      
      await this.cleanupTestEnvironment();
      
    } catch (error) {
      console.error('❌ Test suite failed:', error);
      this.testResults.errors.push({ test: 'setup', error: error.message });
    }
    
    this.printTestSummary();
    return this.testResults;
  }

  async setupTestEnvironment() {
    console.log('🔧 Setting up test environment...');
    
    // Create temporary directory structure
    await fs.mkdir(this.tempDir, { recursive: true });
    await fs.mkdir(path.join(this.tempDir, 'news'), { recursive: true });
    await fs.mkdir(path.join(this.tempDir, 'tests'), { recursive: true });
    await fs.mkdir(path.join(this.tempDir, '.rse-audit'), { recursive: true });
    
    // Create test files
    await this.createTestFiles();
    
    // Initialize auditor with test directory
    this.auditor = new RSEQualityAuditor({
      projectRoot: this.tempDir
    });
    
    console.log('✅ Test environment ready\n');
  }

  async createTestFiles() {
    // Valid news article JSON
    const validNewsArticle = {
      title: 'Test AI News Article',
      description: 'This is a test description for AI news validation testing purposes.',
      url: 'https://example.com/test-article',
      urlToImage: 'https://example.com/test-image.jpg',
      publishedAt: '2024-01-01T12:00:00Z',
      source: {
        name: 'Test News Source',
        url: 'https://example.com'
      },
      category: 'technology',
      sentiment: {
        score: 0.5,
        label: 'positive'
      },
      keywords: ['AI', 'technology', 'test']
    };
    
    await fs.writeFile(
      path.join(this.tempDir, 'news', 'valid-article.json'),
      JSON.stringify(validNewsArticle, null, 2)
    );
    
    // Invalid news article JSON (missing required fields)
    const invalidNewsArticle = {
      title: 'Invalid Article',
      // Missing required fields: description, url, publishedAt, source
      category: 'invalid-category'
    };
    
    await fs.writeFile(
      path.join(this.tempDir, 'news', 'invalid-article.json'),
      JSON.stringify(invalidNewsArticle, null, 2)
    );
    
    // Malformed JSON
    await fs.writeFile(
      path.join(this.tempDir, 'news', 'malformed.json'),
      '{ "title": "Malformed", "invalid": json }'
    );
    
    // Valid Markdown file
    const validMarkdown = `# Test Markdown File

This is a **valid** markdown file for testing.

## Features

- Proper heading format
- No trailing whitespace
- Proper list formatting

### Code Example

\`\`\`javascript
console.log('Hello, World!');
\`\`\`
`;
    
    await fs.writeFile(
      path.join(this.tempDir, 'README.md'),
      validMarkdown
    );
    
    // Invalid Markdown file (style issues)
    const invalidMarkdown = `#Invalid Heading Format   \n\n-No space after list marker\n\nTrailing whitespace here:   \n\n##Another bad heading\n`;
    
    await fs.writeFile(
      path.join(this.tempDir, 'INVALID.md'),
      invalidMarkdown
    );
    
    // Test JavaScript file
    const testJSFile = `/**
 * Test JavaScript file
 */

const testFunction = () => {
  return 'Hello, Test!';
};

module.exports = { testFunction };
`;
    
    await fs.writeFile(
      path.join(this.tempDir, 'test-module.js'),
      testJSFile
    );
    
    // Test file for the test module
    const testFile = `const { testFunction } = require('../test-module.js');

console.log('Testing testFunction...');
const result = testFunction();
if (result === 'Hello, Test!') {
  console.log('✅ Test passed');
} else {
  console.log('❌ Test failed');
  process.exit(1);
}
`;
    
    await fs.writeFile(
      path.join(this.tempDir, 'tests', 'test-module.test.js'),
      testFile
    );
    
    // Package.json for test environment
    const packageJson = {
      name: 'rse-audit-test',
      version: '1.0.0',
      scripts: {
        test: 'node tests/test-module.test.js',
        lint: 'echo "Linting passed"'
      }
    };
    
    await fs.writeFile(
      path.join(this.tempDir, 'package.json'),
      JSON.stringify(packageJson, null, 2)
    );
  }

  async testAuditorInitialization() {
    console.log('🔍 Testing auditor initialization...');
    
    try {
      // Test that auditor initializes correctly
      if (!this.auditor) {
        throw new Error('Auditor not initialized');
      }
      
      // Test directory structure
      const configDir = path.join(this.tempDir, '.rse-audit');
      const stats = await fs.stat(configDir);
      if (!stats.isDirectory()) {
        throw new Error('Config directory not created');
      }
      
      // Test schema loading
      if (!this.auditor.newsSchema || !this.auditor.rssSchema) {
        throw new Error('Schemas not loaded');
      }
      
      this.pass('Auditor initialization');
      
    } catch (error) {
      this.fail('Auditor initialization', error);
    }
  }

  async testSchemaValidation() {
    console.log('🔍 Testing schema validation...');
    
    try {
      // Test valid article validation
      const validFile = path.join(this.tempDir, 'news', 'valid-article.json');
      await this.auditor.validateNewsFile(validFile);
      
      // Test invalid article validation
      const invalidFile = path.join(this.tempDir, 'news', 'invalid-article.json');
      await this.auditor.validateNewsFile(invalidFile);
      
      // Check that validation results are recorded
      if (this.auditor.auditResults.schema.passed === 0 && this.auditor.auditResults.schema.failed === 0) {
        throw new Error('Schema validation not recording results');
      }
      
      this.pass('Schema validation');
      
    } catch (error) {
      this.fail('Schema validation', error);
    }
  }

  async testMarkdownValidation() {
    console.log('🔍 Testing Markdown validation...');
    
    try {
      // Reset markdown results
      this.auditor.auditResults.markdown = { passed: 0, failed: 0, errors: [] };
      
      // Test valid markdown
      const validFile = path.join(this.tempDir, 'README.md');
      await this.auditor.validateMarkdownFile(validFile);
      
      // Test invalid markdown
      const invalidFile = path.join(this.tempDir, 'INVALID.md');
      await this.auditor.validateMarkdownFile(invalidFile);
      
      // Check results
      if (this.auditor.auditResults.markdown.passed === 0 && this.auditor.auditResults.markdown.failed === 0) {
        throw new Error('Markdown validation not recording results');
      }
      
      this.pass('Markdown validation');
      
    } catch (error) {
      this.fail('Markdown validation', error);
    }
  }

  async testTestExecution() {
    console.log('🔍 Testing test execution...');
    
    try {
      // Reset test results
      this.auditor.auditResults.tests = { passed: 0, failed: 0, errors: [] };
      
      // Run tests in test environment
      const originalCwd = process.cwd();
      process.chdir(this.tempDir);
      
      try {
        await this.auditor.runTests();
      } finally {
        process.chdir(originalCwd);
      }
      
      // Check that test execution was attempted
      const hasResults = this.auditor.auditResults.tests.passed > 0 || 
                        this.auditor.auditResults.tests.failed > 0 ||
                        this.auditor.auditResults.tests.errors.length > 0;
      
      if (!hasResults) {
        throw new Error('Test execution not recording results');
      }
      
      this.pass('Test execution');
      
    } catch (error) {
      this.fail('Test execution', error);
    }
  }

  async testLintingExecution() {
    console.log('🔍 Testing linting execution...');
    
    try {
      // Reset linting results
      this.auditor.auditResults.linting = { passed: 0, failed: 0, errors: [] };
      
      // Run linting in test environment
      const originalCwd = process.cwd();
      process.chdir(this.tempDir);
      
      try {
        await this.auditor.runLinting();
      } finally {
        process.chdir(originalCwd);
      }
      
      // Check that linting execution was attempted
      const hasResults = this.auditor.auditResults.linting.passed > 0 || 
                        this.auditor.auditResults.linting.failed > 0 ||
                        this.auditor.auditResults.linting.errors.length > 0;
      
      if (!hasResults) {
        throw new Error('Linting execution not recording results');
      }
      
      this.pass('Linting execution');
      
    } catch (error) {
      this.fail('Linting execution', error);
    }
  }

  async testCoverageCalculation() {
    console.log('🔍 Testing coverage calculation...');
    
    try {
      const sourceFiles = await this.auditor.findSourceFiles();
      const testFiles = await this.auditor.findTestFiles();
      
      const coverage = await this.auditor.calculateCoverage(sourceFiles, testFiles);
      
      // Check coverage structure
      if (typeof coverage.percentage !== 'number' ||
          !Array.isArray(coverage.missing) ||
          typeof coverage.total !== 'number') {
        throw new Error('Invalid coverage calculation structure');
      }
      
      // Coverage should be between 0 and 100
      if (coverage.percentage < 0 || coverage.percentage > 100) {
        throw new Error('Invalid coverage percentage');
      }
      
      this.pass('Coverage calculation');
      
    } catch (error) {
      this.fail('Coverage calculation', error);
    }
  }

  async testSelfAudit() {
    console.log('🔍 Testing self-audit functionality...');
    
    try {
      // Run self-audit
      await this.auditor.selfAudit();
      
      // Check that self-audit results exist
      if (!this.auditor.auditResults.selfAudit) {
        throw new Error('Self-audit results not generated');
      }
      
      const selfAudit = this.auditor.auditResults.selfAudit;
      
      // Check for expected properties
      if (!selfAudit.flakiness && !selfAudit.quality && !selfAudit.error) {
        throw new Error('Self-audit missing expected properties');
      }
      
      this.pass('Self-audit functionality');
      
    } catch (error) {
      this.fail('Self-audit functionality', error);
    }
  }

  async testFullAuditPipeline() {
    console.log('🔍 Testing full audit pipeline...');
    
    try {
      // Run full audit in test environment
      const originalCwd = process.cwd();
      process.chdir(this.tempDir);
      
      let auditResults;
      try {
        auditResults = await this.auditor.runFullAudit();
      } finally {
        process.chdir(originalCwd);
      }
      
      // Check audit results structure
      if (!auditResults || typeof auditResults !== 'object') {
        throw new Error('Invalid audit results');
      }
      
      // Check required properties
      const requiredProps = ['timestamp', 'tests', 'linting', 'schema', 'markdown', 'coverage', 'overall'];
      for (const prop of requiredProps) {
        if (!(prop in auditResults)) {
          throw new Error(`Missing required property: ${prop}`);
        }
      }
      
      // Check overall status
      if (!['passed', 'failed', 'pending'].includes(auditResults.overall)) {
        throw new Error('Invalid overall status');
      }
      
      this.pass('Full audit pipeline');
      
    } catch (error) {
      this.fail('Full audit pipeline', error);
    }
  }

  async testPREventHandling() {
    console.log('🔍 Testing PR event handling...');
    
    try {
      const mockPRData = {
        number: 123,
        title: 'Test PR for RSE Quality Audit',
        body: 'This is a test pull request'
      };
      
      // Run PR event handler in test environment
      const originalCwd = process.cwd();
      process.chdir(this.tempDir);
      
      let prResult;
      try {
        prResult = await this.auditor.handlePREvent(mockPRData);
      } finally {
        process.chdir(originalCwd);
      }
      
      // Check PR result structure
      if (!prResult || !prResult.audit || !prResult.comment || typeof prResult.approved !== 'boolean') {
        throw new Error('Invalid PR event result structure');
      }
      
      // Check comment format
      if (!prResult.comment.includes('RSE Quality Audit Report')) {
        throw new Error('PR comment missing expected content');
      }
      
      this.pass('PR event handling');
      
    } catch (error) {
      this.fail('PR event handling', error);
    }
  }

  async testReportGeneration() {
    console.log('🔍 Testing report generation...');
    
    try {
      // Generate report
      const report = await this.auditor.generateReport();
      
      // Check report structure
      if (!report || !report.summary || !report.recommendations) {
        throw new Error('Invalid report structure');
      }
      
      // Check summary properties
      const summary = report.summary;
      if (typeof summary.total !== 'number' ||
          typeof summary.passed !== 'number' ||
          typeof summary.failed !== 'number' ||
          typeof summary.successRate !== 'number') {
        throw new Error('Invalid summary structure');
      }
      
      // Check that report file was created
      const reportPath = path.join(this.tempDir, '.rse-audit', 'audit-report.json');
      try {
        await fs.access(reportPath);
      } catch (error) {
        throw new Error('Report file not created');
      }
      
      this.pass('Report generation');
      
    } catch (error) {
      this.fail('Report generation', error);
    }
  }

  async testErrorHandling() {
    console.log('🔍 Testing error handling...');
    
    try {
      // Test with non-existent file
      await this.auditor.validateNewsFile('/non/existent/file.json');
      
      // Test with malformed JSON
      const malformedFile = path.join(this.tempDir, 'news', 'malformed.json');
      await this.auditor.validateNewsFile(malformedFile);
      
      // Check that errors were recorded
      if (this.auditor.auditResults.schema.errors.length === 0) {
        throw new Error('Error handling not recording errors');
      }
      
      this.pass('Error handling');
      
    } catch (error) {
      this.fail('Error handling', error);
    }
  }

  async testMissingFiles() {
    console.log('🔍 Testing missing files handling...');
    
    try {
      // Test finding files in non-existent directory
      const tempAuditor = new RSEQualityAuditor({
        projectRoot: '/non/existent/directory'
      });
      
      const newsFiles = await tempAuditor.findNewsFiles();
      const markdownFiles = await tempAuditor.findMarkdownFiles();
      
      // Should return empty arrays, not throw errors
      if (!Array.isArray(newsFiles) || !Array.isArray(markdownFiles)) {
        throw new Error('File finding should return arrays even for missing directories');
      }
      
      this.pass('Missing files handling');
      
    } catch (error) {
      this.fail('Missing files handling', error);
    }
  }

  async testInvalidJSON() {
    console.log('🔍 Testing invalid JSON handling...');
    
    try {
      // Create invalid JSON file
      const invalidJSONPath = path.join(this.tempDir, 'news', 'completely-invalid.json');
      await fs.writeFile(invalidJSONPath, 'This is not JSON at all!');
      
      // Test validation
      await this.auditor.validateNewsFile(invalidJSONPath);
      
      // Should record error, not crash
      const hasJSONError = this.auditor.auditResults.schema.errors.some(
        error => error.file === invalidJSONPath
      );
      
      if (!hasJSONError) {
        throw new Error('Invalid JSON error not recorded');
      }
      
      this.pass('Invalid JSON handling');
      
    } catch (error) {
      this.fail('Invalid JSON handling', error);
    }
  }

  async cleanupTestEnvironment() {
    console.log('\n🧹 Cleaning up test environment...');
    
    try {
      // Remove temporary directory
      await fs.rm(this.tempDir, { recursive: true, force: true });
      console.log('✅ Test environment cleaned up');
    } catch (error) {
      console.warn('Warning: Could not clean up test environment:', error.message);
    }
  }

  pass(testName) {
    console.log(`  ✅ ${testName}`);
    this.testResults.passed++;
  }

  fail(testName, error) {
    console.log(`  ❌ ${testName}: ${error.message}`);
    this.testResults.failed++;
    this.testResults.errors.push({ test: testName, error: error.message });
  }

  printTestSummary() {
    console.log('\n' + '='.repeat(60));
    console.log('📊 RSE AUDIT TEST SUMMARY');
    console.log('='.repeat(60));
    
    const total = this.testResults.passed + this.testResults.failed;
    const successRate = total > 0 ? Math.round((this.testResults.passed / total) * 100) : 100;
    
    console.log(`Total Tests: ${total}`);
    console.log(`Passed: ${this.testResults.passed}`);
    console.log(`Failed: ${this.testResults.failed}`);
    console.log(`Success Rate: ${successRate}%`);
    
    if (this.testResults.errors.length > 0) {
      console.log('\n❌ Failed Tests:');
      this.testResults.errors.forEach((error, i) => {
        console.log(`  ${i + 1}. ${error.test}: ${error.error}`);
      });
    }
    
    console.log('\n' + '='.repeat(60));
    
    if (this.testResults.failed === 0) {
      console.log('🎉 All tests passed!');
    } else {
      console.log('⚠️ Some tests failed. Please review and fix.');
    }
  }
}

// CLI interface
if (require.main === module) {
  const tester = new RSEAuditTester();
  
  tester.runAllTests().then(results => {
    process.exit(results.failed === 0 ? 0 : 1);
  }).catch(error => {
    console.error('Test suite failed:', error);
    process.exit(1);
  });
}

module.exports = { RSEAuditTester };