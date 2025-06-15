#!/usr/bin/env node
/**
 * Service Mapper - Auto-detect existing pipelines, cache layers, and services
 * Part of Dr. NewsForge's AI News Dashboard Enhancement
 */

const fs = require('fs');
const path = require('path');

// Service detection patterns
const SERVICE_PATTERNS = {
  apis: {
    pattern: /\/api\//,
    files: ['route.js', 'index.js', 'handler.js']
  },
  cache: {
    pattern: /(redis|cache|memory)/i,
    files: ['.js', '.ts']
  },
  database: {
    pattern: /(db|database|mongo|postgres|mysql)/i,
    files: ['.js', '.ts', '.json']
  },
  ai: {
    pattern: /(openai|gpt|ai|model|nlp)/i,
    files: ['.js', '.ts']
  },
  ingestion: {
    pattern: /(fetch|ingest|crawler|scraper)/i,
    files: ['.js', '.ts']
  }
};

class ServiceMapper {
  constructor(rootDir) {
    this.rootDir = rootDir;
    this.services = {
      apis: [],
      cache: [],
      database: [],
      ai: [],
      ingestion: [],
      config: [],
      dependencies: []
    };
  }

  async scan() {
    console.log('🔍 Scanning existing services and architecture...');
    
    // Scan package.json for dependencies
    await this.scanDependencies();
    
    // Scan directory structure
    await this.scanDirectory(this.rootDir);
    
    // Analyze configuration files
    await this.scanConfig();
    
    return this.generateReport();
  }

  async scanDependencies() {
    const packagePath = path.join(this.rootDir, 'package.json');
    if (fs.existsSync(packagePath)) {
      const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
      const deps = { ...pkg.dependencies, ...pkg.devDependencies };
      
      this.services.dependencies = Object.keys(deps).map(name => ({
        name,
        version: deps[name],
        type: this.categorizeDependency(name)
      }));
    }
  }

  categorizeDependency(name) {
    if (/react|next|vue|angular/.test(name)) return 'frontend';
    if (/express|fastapi|flask|koa/.test(name)) return 'backend';
    if (/redis|mongo|postgres|mysql/.test(name)) return 'database';
    if (/openai|anthropic|huggingface/.test(name)) return 'ai';
    if (/axios|fetch|request/.test(name)) return 'http';
    if (/tailwind|styled|emotion/.test(name)) return 'styling';
    return 'utility';
  }

  async scanDirectory(dir, depth = 0) {
    if (depth > 3) return; // Limit recursion depth
    
    let items;
    try {
      items = fs.readdirSync(dir);
    } catch (error) {
      console.warn(`⚠️  Cannot read directory: ${dir}`);
      return;
    }
    
    for (const item of items) {
      if (item.startsWith('.') && item !== '.env.local') continue;
      if (item === 'node_modules' || item === '.next') continue; // Skip heavy directories
      
      const fullPath = path.join(dir, item);
      let stat;
      
      try {
        stat = fs.statSync(fullPath);
      } catch (error) {
        continue; // Skip files/dirs we can't access
      }
      
      if (stat.isDirectory()) {
        await this.scanDirectory(fullPath, depth + 1);
      } else if (stat.isFile()) {
        await this.analyzeFile(fullPath);
      }
    }
  }

  async analyzeFile(filePath) {
    const relativePath = path.relative(this.rootDir, filePath);
    let content;
    
    try {
      content = fs.readFileSync(filePath, 'utf8');
    } catch (error) {
      // Skip files that can't be read (binary, permissions, etc.)
      return;
    }
    
    // Check for API routes
    if (relativePath.includes('/api/')) {
      this.services.apis.push({
        path: relativePath,
        type: 'api-route',
        methods: this.extractHttpMethods(content),
        description: this.extractDescription(content)
      });
    }
    
    // Check for AI/ML services
    if (SERVICE_PATTERNS.ai.pattern.test(content) || SERVICE_PATTERNS.ai.pattern.test(relativePath)) {
      this.services.ai.push({
        path: relativePath,
        type: 'ai-service',
        models: this.extractAIModels(content),
        description: this.extractDescription(content)
      });
    }
    
    // Check for data ingestion
    if (SERVICE_PATTERNS.ingestion.pattern.test(content) || SERVICE_PATTERNS.ingestion.pattern.test(relativePath)) {
      this.services.ingestion.push({
        path: relativePath,
        type: 'ingestion-service',
        sources: this.extractDataSources(content),
        description: this.extractDescription(content)
      });
    }
  }

  extractHttpMethods(content) {
    const methods = [];
    if (/export.*async.*function.*GET/m.test(content)) methods.push('GET');
    if (/export.*async.*function.*POST/m.test(content)) methods.push('POST');
    if (/export.*async.*function.*PUT/m.test(content)) methods.push('PUT');
    if (/export.*async.*function.*DELETE/m.test(content)) methods.push('DELETE');
    return methods;
  }

  extractAIModels(content) {
    const models = [];
    if (/gpt-3.5|gpt-4|o4-mini/.test(content)) models.push('OpenAI');
    if (/claude/.test(content)) models.push('Anthropic');
    if (/huggingface|transformers/.test(content)) models.push('HuggingFace');
    return models;
  }

  extractDataSources(content) {
    const sources = [];
    if (/google.*news|newsapi/.test(content)) sources.push('News API');
    if (/youtube|ytdl/.test(content)) sources.push('YouTube');
    if (/rss|feed/.test(content)) sources.push('RSS');
    if (/twitter|x\.com/.test(content)) sources.push('Twitter/X');
    return sources;
  }

  extractDescription(content) {
    // Extract first comment or JSDoc
    const commentMatch = content.match(/\/\*\*?([^*]|\*(?!\/))*\*\//s);
    if (commentMatch) {
      return commentMatch[0].replace(/\/\*\*?|\*\//g, '').replace(/\*/g, '').trim().slice(0, 100);
    }
    
    const lineComment = content.match(/\/\/\s*(.+)/m);
    if (lineComment) {
      return lineComment[1].trim().slice(0, 100);
    }
    
    return 'No description available';
  }

  async scanConfig() {
    const configFiles = ['.env.local', 'next.config.js', 'tailwind.config.js', 'package.json'];
    
    for (const file of configFiles) {
      const filePath = path.join(this.rootDir, file);
      if (fs.existsSync(filePath)) {
        this.services.config.push({
          path: file,
          type: 'config',
          description: `Configuration file: ${file}`
        });
      }
    }
  }

  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalAPIs: this.services.apis.length,
        totalAIServices: this.services.ai.length,
        totalIngestionServices: this.services.ingestion.length,
        totalDependencies: this.services.dependencies.length
      },
      services: this.services,
      recommendations: this.generateRecommendations()
    };
    
    // Write report to file
    const reportPath = path.join(this.rootDir, 'docs', 'architecture.md');
    this.writeArchitectureDoc(report, reportPath);
    
    return report;
  }

  generateRecommendations() {
    const recommendations = [];
    
    // Check for missing services
    if (this.services.cache.length === 0) {
      recommendations.push('Consider adding Redis cache for improved performance');
    }
    
    if (this.services.apis.length < 5) {
      recommendations.push('Expand API endpoints for comprehensive news operations');
    }
    
    if (!this.services.dependencies.find(d => d.name.includes('redis'))) {
      recommendations.push('Add Redis for caching and real-time features');
    }
    
    if (!this.services.dependencies.find(d => d.name.includes('socket'))) {
      recommendations.push('Consider WebSocket support for real-time updates');
    }
    
    return recommendations;
  }

  writeArchitectureDoc(report, filePath) {
    // Ensure docs directory exists
    const docsDir = path.dirname(filePath);
    if (!fs.existsSync(docsDir)) {
      fs.mkdirSync(docsDir, { recursive: true });
    }
    
    const markdown = this.generateMarkdownReport(report);
    fs.writeFileSync(filePath, markdown);
    console.log(`📄 Architecture documentation written to: ${filePath}`);
  }

  generateMarkdownReport(report) {
    return `# AI News Dashboard - Architecture Analysis

*Generated on: ${report.timestamp}*

## Summary

- **Total API Endpoints:** ${report.summary.totalAPIs}
- **AI Services:** ${report.summary.totalAIServices}
- **Data Ingestion Services:** ${report.summary.totalIngestionServices}
- **Dependencies:** ${report.summary.totalDependencies}

## Current Services

### API Endpoints
${report.services.apis.map(api => `- **${api.path}** (${api.methods.join(', ')}) - ${api.description}`).join('\n')}

### AI Services
${report.services.ai.map(ai => `- **${ai.path}** - Models: ${ai.models.join(', ')} - ${ai.description}`).join('\n')}

### Data Ingestion
${report.services.ingestion.map(ing => `- **${ing.path}** - Sources: ${ing.sources.join(', ')} - ${ing.description}`).join('\n')}

### Dependencies by Category
${this.groupDependenciesByType(report.services.dependencies)}

## Recommendations

${report.recommendations.map(rec => `- ${rec}`).join('\n')}

## Next Steps

1. Implement missing cache layer (Redis)
2. Add real-time WebSocket support
3. Expand NLP capabilities
4. Implement user personalization
5. Add comprehensive monitoring

---
*This document is auto-generated by the Service Mapper*
`;
  }

  groupDependenciesByType(dependencies) {
    const grouped = dependencies.reduce((acc, dep) => {
      if (!acc[dep.type]) acc[dep.type] = [];
      acc[dep.type].push(dep);
      return acc;
    }, {});
    
    return Object.entries(grouped)
      .map(([type, deps]) => `\n**${type.toUpperCase()}:**\n${deps.map(d => `  - ${d.name} (${d.version})`).join('\n')}`)
      .join('\n');
  }
}

// CLI execution
if (require.main === module) {
  const rootDir = process.argv[2] || process.cwd();
  const mapper = new ServiceMapper(rootDir);
  
  mapper.scan().then(report => {
    console.log('\n🎯 Service Mapping Complete!');
    console.log(`Found ${report.summary.totalAPIs} APIs, ${report.summary.totalAIServices} AI services`);
    console.log('📊 Full report saved to docs/architecture.md');
  }).catch(console.error);
}

module.exports = ServiceMapper;