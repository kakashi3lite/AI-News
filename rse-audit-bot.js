#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

/**
 * RSE Quality Audit Bot
 * Automated code quality assessment tool for Research Software Engineering projects
 */
class RSEQualityAuditor {
  constructor(options = {}) {
    this.projectRoot = options.projectRoot || process.cwd();
    this.sourceDir = options.sourceDir || 'src';
    this.testsDir = options.testsDir || 'tests';
    this.docsDir = options.docsDir || 'docs';
    this.configFiles = options.configFiles || [
      'package.json',
      'requirements.txt',
      'setup.py',
      'Cargo.toml',
      'go.mod'
    ];
    
    // Quality thresholds
    this.thresholds = {
      testCoverage: options.testCoverage || 80,
      codeComplexity: options.codeComplexity || 10,
      documentationCoverage: options.documentationCoverage || 70,
      maintainabilityIndex: options.maintainabilityIndex || 70
    };
    
    // File patterns to analyze
    this.patterns = {
      source: /\.(js|ts|py|rs|go|java|cpp|c|h)$/,
      test: /\.(test|spec)\.(js|ts|py|rs|go)$/,
      docs: /\.(md|rst|txt)$/,
      config: /\.(json|yaml|yml|toml|ini)$/
    };
  }

  /**
   * Run comprehensive quality audit
   */
  async audit() {
    console.log('🔍 Starting RSE Quality Audit...');
    
    const results = {
      timestamp: new Date().toISOString(),
      project: await this.getProjectInfo(),
      metrics: {},
      recommendations: [],
      score: 0
    };

    try {
      // Core quality metrics
      results.metrics.testCoverage = await this.analyzeTestCoverage();
      results.metrics.codeQuality = await this.analyzeCodeQuality();
      results.metrics.documentation = await this.analyzeDocumentation();
      results.metrics.dependencies = await this.analyzeDependencies();
      results.metrics.security = await this.analyzeSecurityIssues();
      results.metrics.performance = await this.analyzePerformance();
      
      // Calculate overall score
      results.score = this.calculateOverallScore(results.metrics);
      
      // Generate recommendations
      results.recommendations = this.generateRecommendations(results.metrics);
      
      // Output results
      await this.outputResults(results);
      
      console.log(`✅ Audit completed. Overall score: ${results.score}/100`);
      return results;
      
    } catch (error) {
      console.error('❌ Audit failed:', error.message);
      throw error;
    }
  }

  /**
   * Get basic project information
   */
  async getProjectInfo() {
    const info = {
      name: path.basename(this.projectRoot),
      path: this.projectRoot,
      language: 'unknown',
      framework: 'unknown',
      size: 0
    };

    try {
      // Detect primary language and framework
      const files = await this.getAllFiles();
      info.size = files.length;
      
      const extensions = files.map(f => path.extname(f)).filter(Boolean);
      const extCounts = extensions.reduce((acc, ext) => {
        acc[ext] = (acc[ext] || 0) + 1;
        return acc;
      }, {});
      
      const primaryExt = Object.keys(extCounts).reduce((a, b) => 
        extCounts[a] > extCounts[b] ? a : b, '.txt');
      
      info.language = this.getLanguageFromExtension(primaryExt);
      info.framework = await this.detectFramework();
      
    } catch (error) {
      console.warn('Warning: Could not gather complete project info:', error.message);
    }

    return info;
  }

  /**
   * Analyze test coverage
   */
  async analyzeTestCoverage() {
    console.log('📊 Analyzing test coverage...');
    
    const coverage = {
      percentage: 0,
      testedFiles: 0,
      totalFiles: 0,
      missingTests: [],
      testFiles: []
    };

    try {
      const sourceFiles = await this.getSourceFiles();
      const testFiles = await this.getTestFiles();
      
      coverage.totalFiles = sourceFiles.length;
      coverage.testFiles = testFiles;
      
      // Check which source files have corresponding tests
      const testedFiles = [];
      for (const sourceFile of sourceFiles) {
        if (await this.hasCorrespondingTest(sourceFile, testFiles)) {
          testedFiles.push(sourceFile);
        } else {
          coverage.missingTests.push(sourceFile);
        }
      }
      
      coverage.testedFiles = testedFiles.length;
      coverage.percentage = sourceFiles.length > 0 ? 
        Math.round((testedFiles.length / sourceFiles.length) * 100) : 0;
      
      // Try to get actual coverage from tools
      const actualCoverage = await this.getActualCoverage();
      if (actualCoverage !== null) {
        coverage.percentage = actualCoverage;
      }
      
    } catch (error) {
      console.warn('Warning: Test coverage analysis failed:', error.message);
    }

    return coverage;
  }

  /**
   * Analyze code quality metrics
   */
  async analyzeCodeQuality() {
    console.log('🔧 Analyzing code quality...');
    
    const quality = {
      complexity: 0,
      maintainability: 0,
      duplication: 0,
      linting: {
        errors: 0,
        warnings: 0,
        issues: []
      }
    };

    try {
      const sourceFiles = await this.getSourceFiles();
      
      // Calculate cyclomatic complexity
      quality.complexity = await this.calculateComplexity(sourceFiles);
      
      // Run linting if available
      quality.linting = await this.runLinting();
      
      // Calculate maintainability index
      quality.maintainability = await this.calculateMaintainability(sourceFiles);
      
      // Detect code duplication
      quality.duplication = await this.detectDuplication(sourceFiles);
      
    } catch (error) {
      console.warn('Warning: Code quality analysis failed:', error.message);
    }

    return quality;
  }

  /**
   * Analyze documentation coverage
   */
  async analyzeDocumentation() {
    console.log('📚 Analyzing documentation...');
    
    const docs = {
      coverage: 0,
      files: [],
      missingDocs: [],
      readmeExists: false,
      apiDocsExists: false
    };

    try {
      // Check for README
      docs.readmeExists = await this.fileExists('README.md') || 
                         await this.fileExists('README.rst') ||
                         await this.fileExists('README.txt');
      
      // Find documentation files
      docs.files = await this.getDocumentationFiles();
      
      // Check for API documentation
      docs.apiDocsExists = docs.files.some(f => 
        f.toLowerCase().includes('api') || f.toLowerCase().includes('reference'));
      
      // Calculate documentation coverage
      const sourceFiles = await this.getSourceFiles();
      const documentedFunctions = await this.countDocumentedFunctions(sourceFiles);
      const totalFunctions = await this.countTotalFunctions(sourceFiles);
      
      docs.coverage = totalFunctions > 0 ? 
        Math.round((documentedFunctions / totalFunctions) * 100) : 0;
      
    } catch (error) {
      console.warn('Warning: Documentation analysis failed:', error.message);
    }

    return docs;
  }

  /**
   * Analyze dependencies and security
   */
  async analyzeDependencies() {
    console.log('📦 Analyzing dependencies...');
    
    const deps = {
      total: 0,
      outdated: [],
      vulnerable: [],
      licenses: {},
      devDependencies: 0
    };

    try {
      // Check package.json for Node.js projects
      if (await this.fileExists('package.json')) {
        const packageJson = JSON.parse(await fs.readFile(
          path.join(this.projectRoot, 'package.json'), 'utf8'));
        
        deps.total = Object.keys(packageJson.dependencies || {}).length;
        deps.devDependencies = Object.keys(packageJson.devDependencies || {}).length;
        
        // Check for outdated packages
        deps.outdated = await this.checkOutdatedPackages();
      }
      
      // Check requirements.txt for Python projects
      if (await this.fileExists('requirements.txt')) {
        const requirements = await fs.readFile(
          path.join(this.projectRoot, 'requirements.txt'), 'utf8');
        deps.total = requirements.split('\n').filter(line => 
          line.trim() && !line.startsWith('#')).length;
      }
      
    } catch (error) {
      console.warn('Warning: Dependency analysis failed:', error.message);
    }

    return deps;
  }

  /**
   * Analyze security issues
   */
  async analyzeSecurityIssues() {
    console.log('🔒 Analyzing security...');
    
    const security = {
      vulnerabilities: [],
      secrets: [],
      permissions: [],
      score: 100
    };

    try {
      // Check for common security issues
      security.secrets = await this.scanForSecrets();
      security.vulnerabilities = await this.scanForVulnerabilities();
      
      // Calculate security score
      security.score = Math.max(0, 100 - 
        (security.secrets.length * 10) - 
        (security.vulnerabilities.length * 15));
      
    } catch (error) {
      console.warn('Warning: Security analysis failed:', error.message);
    }

    return security;
  }

  /**
   * Analyze performance characteristics
   */
  async analyzePerformance() {
    console.log('⚡ Analyzing performance...');
    
    const performance = {
      bundleSize: 0,
      loadTime: 0,
      memoryUsage: 0,
      optimizations: []
    };

    try {
      // Estimate bundle size for web projects
      if (await this.fileExists('package.json')) {
        performance.bundleSize = await this.estimateBundleSize();
      }
      
      // Check for performance optimizations
      performance.optimizations = await this.checkPerformanceOptimizations();
      
    } catch (error) {
      console.warn('Warning: Performance analysis failed:', error.message);
    }

    return performance;
  }

  /**
   * Calculate overall quality score
   */
  calculateOverallScore(metrics) {
    const weights = {
      testCoverage: 0.25,
      codeQuality: 0.25,
      documentation: 0.20,
      security: 0.20,
      dependencies: 0.10
    };

    let score = 0;
    
    // Test coverage score
    score += (metrics.testCoverage.percentage / 100) * weights.testCoverage * 100;
    
    // Code quality score (inverse of complexity, plus maintainability)
    const complexityScore = Math.max(0, 100 - (metrics.codeQuality.complexity * 5));
    const qualityScore = (complexityScore + metrics.codeQuality.maintainability) / 2;
    score += (qualityScore / 100) * weights.codeQuality * 100;
    
    // Documentation score
    score += (metrics.documentation.coverage / 100) * weights.documentation * 100;
    
    // Security score
    score += (metrics.security.score / 100) * weights.security * 100;
    
    // Dependencies score (fewer outdated = better)
    const depScore = Math.max(0, 100 - (metrics.dependencies.outdated.length * 10));
    score += (depScore / 100) * weights.dependencies * 100;
    
    return Math.round(score);
  }

  /**
   * Generate recommendations based on metrics
   */
  generateRecommendations(metrics) {
    const recommendations = [];

    // Test coverage recommendations
    if (metrics.testCoverage.percentage < this.thresholds.testCoverage) {
      recommendations.push({
        category: 'Testing',
        priority: 'high',
        message: `Test coverage is ${metrics.testCoverage.percentage}%. Add tests for: ${metrics.testCoverage.missingTests.slice(0, 3).join(', ')}`,
        action: 'Add unit tests for uncovered files'
      });
    }

    // Code quality recommendations
    if (metrics.codeQuality.complexity > this.thresholds.codeComplexity) {
      recommendations.push({
        category: 'Code Quality',
        priority: 'medium',
        message: `Code complexity is high (${metrics.codeQuality.complexity}). Consider refactoring complex functions.`,
        action: 'Refactor complex functions and reduce cyclomatic complexity'
      });
    }

    // Documentation recommendations
    if (metrics.documentation.coverage < this.thresholds.documentationCoverage) {
      recommendations.push({
        category: 'Documentation',
        priority: 'medium',
        message: `Documentation coverage is ${metrics.documentation.coverage}%. Add docstrings and comments.`,
        action: 'Add documentation for functions and classes'
      });
    }

    // Security recommendations
    if (metrics.security.secrets.length > 0) {
      recommendations.push({
        category: 'Security',
        priority: 'high',
        message: `Found ${metrics.security.secrets.length} potential secrets in code.`,
        action: 'Remove hardcoded secrets and use environment variables'
      });
    }

    // Dependencies recommendations
    if (metrics.dependencies.outdated.length > 0) {
      recommendations.push({
        category: 'Dependencies',
        priority: 'low',
        message: `${metrics.dependencies.outdated.length} dependencies are outdated.`,
        action: 'Update outdated dependencies'
      });
    }

    return recommendations;
  }

  /**
   * Helper method to calculate code complexity
   */
  async calculateComplexity(files) {
    let totalComplexity = 0;
    
    for (const file of files) {
      try {
        const content = await fs.readFile(file, 'utf8');
        totalComplexity += this.calculateFileComplexity(content);
      } catch (error) {
        console.warn(`Warning: Could not analyze complexity for ${file}:`, error.message);
      }
    }
    
    return files.length > 0 ? Math.round(totalComplexity / files.length) : 0;
  }

  /**
   * Calculate complexity for a single file
   */
  calculateFileComplexity(content) {
    // Simple cyclomatic complexity calculation
    const complexityPatterns = [
      /\bif\b/g,
      /\belse\b/g,
      /\bwhile\b/g,
      /\bfor\b/g,
      /\bswitch\b/g,
      /\bcase\b/g,
      /\bcatch\b/g,
      /\b&&\b/g,
      /\b\|\|\b/g,
      /\?.*:/g
    ];
    
    let complexity = 1; // Base complexity
    
    complexityPatterns.forEach(regex => {
      const matches = content.match(regex);
      if (matches) {
        complexity += matches.length;
      }
    });
    
    return complexity;
  }

  /**
   * Check if a file has corresponding tests
   */
  async checkTestCoverage(filePath) {
    const fileName = path.basename(filePath, path.extname(filePath));
    const testPatterns = [
      path.join(this.testsDir, `${fileName}.test.js`),
      path.join(this.testsDir, `test-${fileName}.js`),
      path.join(this.testsDir, `${fileName}.spec.js`)
    ];
    
    for (const testPath of testPatterns) {
      if (await this.fileExists(testPath)) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * Calculate test coverage using actual coverage tools
   */
  async calculateCoverage() {
    try {
      // Get all test files
      const testFiles = await this.getTestFiles();
      const sourceFiles = await this.getSourceFiles();
      
      if (testFiles.length === 0) {
        return { percentage: 0, testedModules: [], untestedModules: sourceFiles };
      }
      
      // Analyze which modules are tested
      const testedModules = new Set();
      
      for (const testFile of testFiles) {
        const content = await fs.readFile(testFile, 'utf8');
        // Look for require/import statements
        const requires = content.match(/require\(['"](.*)['"]/\)/g) || [];
        const importsFrom = content.match(/from\s+['"]([^'"]*)['"])/g) || [];
        const importsStmt = content.match(/import\s+['"]([^'"]*)['"])/g) || [];
        const imports = [...importsFrom, ...importsStmt];
        
        [...requires, ...imports].forEach(stmt => {
          const match = stmt.match(/['"](.*)['"]/)?.[1];
          if (match && match.startsWith('.')) {
            testedModules.add(path.resolve(path.dirname(testFile), match));
          }
        });
      }
      
      // Calculate coverage percentage
      const coverage = sourceFiles.length > 0 ? 
        (testedModules.size / sourceFiles.length) * 100 : 0;
      
      return {
        percentage: Math.round(coverage),
        testedModules: Array.from(testedModules),
        untestedModules: sourceFiles.filter(f => !testedModules.has(f))
      };
      
    } catch (error) {
      console.warn('Warning: Coverage calculation failed:', error.message);
      return { percentage: 0, testedModules: [], untestedModules: [] };
    }
  }

  /**
   * Get all files in the project
   */
  async getAllFiles(dir = this.projectRoot, files = []) {
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory() && !this.shouldIgnoreDirectory(entry.name)) {
          await this.getAllFiles(fullPath, files);
        } else if (entry.isFile()) {
          files.push(fullPath);
        }
      }
    } catch (error) {
      console.warn(`Warning: Could not read directory ${dir}:`, error.message);
    }
    
    return files;
  }

  /**
   * Get source files
   */
  async getSourceFiles() {
    const allFiles = await this.getAllFiles();
    return allFiles.filter(file => 
      this.patterns.source.test(file) && 
      !this.patterns.test.test(file) &&
      !file.includes('node_modules') &&
      !file.includes('.git')
    );
  }

  /**
   * Get test files
   */
  async getTestFiles() {
    const allFiles = await this.getAllFiles();
    return allFiles.filter(file => this.patterns.test.test(file));
  }

  /**
   * Get documentation files
   */
  async getDocumentationFiles() {
    const allFiles = await this.getAllFiles();
    return allFiles.filter(file => this.patterns.docs.test(file));
  }

  /**
   * Check if a file exists
   */
  async fileExists(filePath) {
    try {
      await fs.access(path.join(this.projectRoot, filePath));
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Check if directory should be ignored
   */
  shouldIgnoreDirectory(dirName) {
    const ignoreDirs = [
      'node_modules', '.git', '.svn', '.hg',
      'build', 'dist', 'target', '__pycache__',
      '.pytest_cache', '.coverage', '.nyc_output'
    ];
    return ignoreDirs.includes(dirName) || dirName.startsWith('.');
  }

  /**
   * Detect primary framework
   */
  async detectFramework() {
    try {
      if (await this.fileExists('package.json')) {
        const packageJson = JSON.parse(await fs.readFile(
          path.join(this.projectRoot, 'package.json'), 'utf8'));
        
        const deps = { ...packageJson.dependencies, ...packageJson.devDependencies };
        
        if (deps.react) return 'React';
        if (deps.vue) return 'Vue';
        if (deps.angular) return 'Angular';
        if (deps.express) return 'Express';
        if (deps.next) return 'Next.js';
        if (deps.nuxt) return 'Nuxt.js';
        
        return 'Node.js';
      }
      
      if (await this.fileExists('requirements.txt') || await this.fileExists('setup.py')) {
        return 'Python';
      }
      
      if (await this.fileExists('Cargo.toml')) {
        return 'Rust';
      }
      
      if (await this.fileExists('go.mod')) {
        return 'Go';
      }
      
    } catch (error) {
      console.warn('Warning: Framework detection failed:', error.message);
    }
    
    return 'Unknown';
  }

  /**
   * Get language from file extension
   */
  getLanguageFromExtension(ext) {
    const languageMap = {
      '.js': 'JavaScript',
      '.ts': 'TypeScript',
      '.py': 'Python',
      '.rs': 'Rust',
      '.go': 'Go',
      '.java': 'Java',
      '.cpp': 'C++',
      '.c': 'C',
      '.h': 'C/C++'
    };
    
    return languageMap[ext] || 'Unknown';
  }

  /**
   * Check if source file has corresponding test
   */
  async hasCorrespondingTest(sourceFile, testFiles) {
    const baseName = path.basename(sourceFile, path.extname(sourceFile));
    
    return testFiles.some(testFile => {
      const testBaseName = path.basename(testFile);
      return testBaseName.includes(baseName);
    });
  }

  /**
   * Get actual coverage from coverage tools
   */
  async getActualCoverage() {
    try {
      // Try to run coverage tools
      if (await this.fileExists('package.json')) {
        try {
          const result = execSync('npm run coverage --silent', { 
            cwd: this.projectRoot,
            encoding: 'utf8',
            timeout: 30000
          });
          
          // Parse coverage output
          const coverageMatch = result.match(/All files.*?(\d+\.?\d*)%/);
          if (coverageMatch) {
            return parseFloat(coverageMatch[1]);
          }
        } catch (error) {
          // Coverage command might not exist
        }
      }
      
      return null;
    } catch (error) {
      return null;
    }
  }

  /**
   * Run linting tools
   */
  async runLinting() {
    const linting = {
      errors: 0,
      warnings: 0,
      issues: []
    };

    try {
      if (await this.fileExists('package.json')) {
        try {
          execSync('npm run lint --silent', { 
            cwd: this.projectRoot,
            encoding: 'utf8',
            timeout: 30000
          });
        } catch (error) {
          // Parse linting output
          const output = error.stdout || error.stderr || '';
          const errorMatches = output.match(/error/gi) || [];
          const warningMatches = output.match(/warning/gi) || [];
          
          linting.errors = errorMatches.length;
          linting.warnings = warningMatches.length;
          linting.issues = output.split('\n').filter(line => 
            line.includes('error') || line.includes('warning')).slice(0, 10);
        }
      }
    } catch (error) {
      console.warn('Warning: Linting analysis failed:', error.message);
    }

    return linting;
  }

  /**
   * Calculate maintainability index
   */
  async calculateMaintainability(files) {
    let totalMaintainability = 0;
    
    for (const file of files) {
      try {
        const content = await fs.readFile(file, 'utf8');
        const complexity = this.calculateFileComplexity(content);
        const linesOfCode = content.split('\n').length;
        
        // Simplified maintainability index calculation
        const maintainability = Math.max(0, 100 - (complexity * 2) - (linesOfCode / 10));
        totalMaintainability += maintainability;
      } catch (error) {
        console.warn(`Warning: Could not analyze maintainability for ${file}:`, error.message);
      }
    }
    
    return files.length > 0 ? Math.round(totalMaintainability / files.length) : 0;
  }

  /**
   * Detect code duplication
   */
  async detectDuplication(files) {
    // Simple duplication detection based on similar lines
    const lineHashes = new Map();
    let duplicatedLines = 0;
    let totalLines = 0;
    
    for (const file of files) {
      try {
        const content = await fs.readFile(file, 'utf8');
        const lines = content.split('\n').filter(line => line.trim().length > 10);
        
        totalLines += lines.length;
        
        for (const line of lines) {
          const hash = this.simpleHash(line.trim());
          if (lineHashes.has(hash)) {
            duplicatedLines++;
          } else {
            lineHashes.set(hash, true);
          }
        }
      } catch (error) {
        console.warn(`Warning: Could not analyze duplication for ${file}:`, error.message);
      }
    }
    
    return totalLines > 0 ? Math.round((duplicatedLines / totalLines) * 100) : 0;
  }

  /**
   * Count documented functions
   */
  async countDocumentedFunctions(files) {
    let documentedCount = 0;
    
    for (const file of files) {
      try {
        const content = await fs.readFile(file, 'utf8');
        
        // Look for functions with documentation
        const functionMatches = content.match(/\/\*\*[\s\S]*?\*\/[\s\S]*?function|function[\s\S]*?\/\*\*[\s\S]*?\*\//g) || [];
        const methodMatches = content.match(/\/\*\*[\s\S]*?\*\/[\s\S]*?\w+\s*\([^)]*\)\s*{|\w+\s*\([^)]*\)\s*{[\s\S]*?\/\*\*[\s\S]*?\*\//g) || [];
        
        documentedCount += functionMatches.length + methodMatches.length;
      } catch (error) {
        console.warn(`Warning: Could not count documented functions for ${file}:`, error.message);
      }
    }
    
    return documentedCount;
  }

  /**
   * Count total functions
   */
  async countTotalFunctions(files) {
    let totalCount = 0;
    
    for (const file of files) {
      try {
        const content = await fs.readFile(file, 'utf8');
        
        // Count function declarations
        const functionMatches = content.match(/function\s+\w+|\w+\s*\([^)]*\)\s*{|\w+\s*=\s*\([^)]*\)\s*=>/g) || [];
        totalCount += functionMatches.length;
      } catch (error) {
        console.warn(`Warning: Could not count total functions for ${file}:`, error.message);
      }
    }
    
    return totalCount;
  }

  /**
   * Check for outdated packages
   */
  async checkOutdatedPackages() {
    try {
      if (await this.fileExists('package.json')) {
        const result = execSync('npm outdated --json', { 
          cwd: this.projectRoot,
          encoding: 'utf8',
          timeout: 30000
        });
        
        const outdated = JSON.parse(result);
        return Object.keys(outdated);
      }
    } catch (error) {
      // npm outdated returns non-zero exit code when packages are outdated
      try {
        const outdated = JSON.parse(error.stdout || '{}');
        return Object.keys(outdated);
      } catch {
        return [];
      }
    }
    
    return [];
  }

  /**
   * Scan for potential secrets
   */
  async scanForSecrets() {
    const secrets = [];
    const secretPatterns = [
      /api[_-]?key[\s]*[:=][\s]*['"]([^'"]+)['"]/gi,
      /secret[_-]?key[\s]*[:=][\s]*['"]([^'"]+)['"]/gi,
      /password[\s]*[:=][\s]*['"]([^'"]+)['"]/gi,
      /token[\s]*[:=][\s]*['"]([^'"]+)['"]/gi,
      /['"]([A-Za-z0-9]{32,})['"]/g // Generic long strings
    ];
    
    const sourceFiles = await this.getSourceFiles();
    
    for (const file of sourceFiles) {
      try {
        const content = await fs.readFile(file, 'utf8');
        
        for (const pattern of secretPatterns) {
          const matches = content.match(pattern);
          if (matches) {
            secrets.push({
              file: path.relative(this.projectRoot, file),
              matches: matches.slice(0, 3) // Limit to first 3 matches
            });
          }
        }
      } catch (error) {
        console.warn(`Warning: Could not scan for secrets in ${file}:`, error.message);
      }
    }
    
    return secrets;
  }

  /**
   * Scan for vulnerabilities
   */
  async scanForVulnerabilities() {
    const vulnerabilities = [];
    
    try {
      if (await this.fileExists('package.json')) {
        const result = execSync('npm audit --json', { 
          cwd: this.projectRoot,
          encoding: 'utf8',
          timeout: 30000
        });
        
        const audit = JSON.parse(result);
        if (audit.vulnerabilities) {
          Object.entries(audit.vulnerabilities).forEach(([pkg, vuln]) => {
            vulnerabilities.push({
              package: pkg,
              severity: vuln.severity,
              title: vuln.title
            });
          });
        }
      }
    } catch (error) {
      // npm audit might return non-zero exit code
      try {
        const audit = JSON.parse(error.stdout || '{}');
        if (audit.vulnerabilities) {
          Object.entries(audit.vulnerabilities).forEach(([pkg, vuln]) => {
            vulnerabilities.push({
              package: pkg,
              severity: vuln.severity,
              title: vuln.title
            });
          });
        }
      } catch {
        // Ignore parsing errors
      }
    }
    
    return vulnerabilities;
  }

  /**
   * Estimate bundle size
   */
  async estimateBundleSize() {
    try {
      const sourceFiles = await this.getSourceFiles();
      let totalSize = 0;
      
      for (const file of sourceFiles) {
        const stats = await fs.stat(file);
        totalSize += stats.size;
      }
      
      return Math.round(totalSize / 1024); // Return size in KB
    } catch (error) {
      console.warn('Warning: Bundle size estimation failed:', error.message);
      return 0;
    }
  }

  /**
   * Check for performance optimizations
   */
  async checkPerformanceOptimizations() {
    const optimizations = [];
    
    try {
      if (await this.fileExists('package.json')) {
        const packageJson = JSON.parse(await fs.readFile(
          path.join(this.projectRoot, 'package.json'), 'utf8'));
        
        // Check for common optimization tools
        const deps = { ...packageJson.dependencies, ...packageJson.devDependencies };
        
        if (deps.webpack) optimizations.push('Webpack bundling');
        if (deps.babel) optimizations.push('Babel transpilation');
        if (deps.terser || deps['uglify-js']) optimizations.push('Code minification');
        if (deps['compression-webpack-plugin']) optimizations.push('Gzip compression');
        if (deps.workbox) optimizations.push('Service worker caching');
      }
    } catch (error) {
      console.warn('Warning: Performance optimization check failed:', error.message);
    }
    
    return optimizations;
  }

  /**
   * Simple hash function for duplicate detection
   */
  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }

  /**
   * Output audit results
   */
  async outputResults(results) {
    const outputFile = path.join(this.projectRoot, 'rse-audit-report.json');
    
    try {
      await fs.writeFile(outputFile, JSON.stringify(results, null, 2));
      console.log(`📄 Audit report saved to: ${outputFile}`);
      
      // Also output summary to console
      this.printSummary(results);
      
    } catch (error) {
      console.error('❌ Failed to save audit report:', error.message);
      this.printSummary(results);
    }
  }

  /**
   * Print audit summary to console
   */
  printSummary(results) {
    console.log('\n' + '='.repeat(60));
    console.log('🎯 RSE QUALITY AUDIT SUMMARY');
    console.log('='.repeat(60));
    
    console.log(`\n📊 Overall Score: ${results.score}/100`);
    
    console.log('\n📈 Metrics:');
    console.log(`  • Test Coverage: ${results.metrics.testCoverage.percentage}%`);
    console.log(`  • Code Complexity: ${results.metrics.codeQuality.complexity}`);
    console.log(`  • Documentation: ${results.metrics.documentation.coverage}%`);
    console.log(`  • Security Score: ${results.metrics.security.score}/100`);
    
    if (results.recommendations.length > 0) {
      console.log('\n💡 Top Recommendations:');
      results.recommendations.slice(0, 5).forEach((rec, index) => {
        const priority = rec.priority === 'high' ? '🔴' : 
                        rec.priority === 'medium' ? '🟡' : '🟢';
        console.log(`  ${index + 1}. ${priority} ${rec.message}`);
      });
    }
    
    console.log('\n' + '='.repeat(60));
  }

  /**
   * Run tests
   */
  async test() {
    console.log('🧪 Running RSE Quality Auditor tests...');
    
    try {
      // Basic functionality tests
      await this.testBasicFunctionality();
      await this.testFileDetection();
      await this.testMetricsCalculation();
      
      console.log('✅ All tests passed!');
      return true;
      
    } catch (error) {
      console.error('❌ Tests failed:', error.message);
      return false;
    }
  }

  /**
   * Test basic functionality
   */
  async testBasicFunctionality() {
    console.log('  Testing basic functionality...');
    
    // Test project info gathering
    const projectInfo = await this.getProjectInfo();
    if (!projectInfo.name) {
      throw new Error('Failed to get project name');
    }
    
    console.log('    ✓ Project info gathering');
  }

  /**
   * Test file detection
   */
  async testFileDetection() {
    console.log('  Testing file detection...');
    
    const sourceFiles = await this.getSourceFiles();
    const testFiles = await this.getTestFiles();
    const docFiles = await this.getDocumentationFiles();
    
    console.log(`    ✓ Found ${sourceFiles.length} source files`);
    console.log(`    ✓ Found ${testFiles.length} test files`);
    console.log(`    ✓ Found ${docFiles.length} documentation files`);
  }

  /**
   * Test metrics calculation
   */
  async testMetricsCalculation() {
    console.log('  Testing metrics calculation...');
    
    const sourceFiles = await this.getSourceFiles();
    if (sourceFiles.length > 0) {
      const complexity = await this.calculateComplexity(sourceFiles.slice(0, 1));
      if (typeof complexity !== 'number') {
        throw new Error('Complexity calculation failed');
      }
      console.log('    ✓ Complexity calculation');
    }
    
    const coverage = await this.calculateCoverage();
    if (typeof coverage.percentage !== 'number') {
      throw new Error('Coverage calculation failed');
    }
    console.log('    ✓ Coverage calculation');
  }

  /**
   * Run linting
   */
  async lint() {
    console.log('🔍 Running RSE Quality Auditor linting...');
    
    try {
      const sourceFiles = await this.getSourceFiles();
      let issues = 0;
      
      for (const file of sourceFiles) {
        const fileIssues = await this.lintFile(file);
        issues += fileIssues.length;
        
        if (fileIssues.length > 0) {
          console.log(`  ${file}:`);
          fileIssues.forEach(issue => {
            console.log(`    ${issue.line}: ${issue.message}`);
          });
        }
      }
      
      if (issues === 0) {
        console.log('✅ No linting issues found!');
      } else {
        console.log(`⚠️  Found ${issues} linting issues`);
      }
      
      return issues === 0;
      
    } catch (error) {
      console.error('❌ Linting failed:', error.message);
      return false;
    }
  }

  /**
   * Lint a single file
   */
  async lintFile(filePath) {
    const issues = [];
    
    try {
      const content = await fs.readFile(filePath, 'utf8');
      const lines = content.split('\n');
      
      lines.forEach((line, index) => {
        // Basic linting rules
        if (line.length > 120) {
          issues.push({
            line: index + 1,
            message: 'Line too long (>120 characters)'
          });
        }
        
        if (line.includes('console.log') && !filePath.includes('test')) {
          issues.push({
            line: index + 1,
            message: 'Avoid console.log in production code'
          });
        }
        
        if (line.includes('TODO') || line.includes('FIXME')) {
          issues.push({
            line: index + 1,
            message: 'TODO/FIXME comment found'
          });
        }
      });
      
    } catch (error) {
      console.warn(`Warning: Could not lint ${filePath}:`, error.message);
    }
    
    return issues;
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0] || 'audit';
  
  const auditor = new RSEQualityAuditor();
  
  switch (command) {
    case 'audit':
      auditor.audit().catch(error => {
        console.error('Audit failed:', error.message);
        process.exit(1);
      });
      break;
      
    case 'test':
      auditor.test().then(success => {
        process.exit(success ? 0 : 1);
      });
      break;
      
    case 'lint':
      auditor.lint().then(success => {
        process.exit(success ? 0 : 1);
      });
      break;
      
    default:
      console.log('Usage: node rse-audit-bot.js [audit|test|lint]');
      process.exit(1);
  }
}

module.exports = RSEQualityAuditor;