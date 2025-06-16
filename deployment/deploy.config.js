// MVP Deployment Configuration for AI News Dashboard
// Dr. Phoenix "SoloSprint" Vega - Context-Aware Solo Founder Setup

const deploymentConfig = {
  // Environment Configuration
  environments: {
    development: {
      name: 'Development',
      url: 'http://localhost:3000',
      apiUrl: 'http://localhost:3001/api',
      features: {
        experimentationEngine: true,
        contextAwareness: true,
        socialFeatures: true,
        voiceSearch: true,
        realTimeCollaboration: false,
        advancedAI: true,
        analytics: true
      },
      database: {
        type: 'sqlite',
        path: './data/dev.db'
      },
      redis: {
        enabled: false
      },
      monitoring: {
        enabled: false
      }
    },
    staging: {
      name: 'Staging',
      url: 'https://staging-ai-news.vercel.app',
      apiUrl: 'https://staging-api-ai-news.vercel.app/api',
      features: {
        experimentationEngine: true,
        contextAwareness: true,
        socialFeatures: true,
        voiceSearch: true,
        realTimeCollaboration: true,
        advancedAI: true,
        analytics: true
      },
      database: {
        type: 'postgresql',
        url: process.env.STAGING_DATABASE_URL
      },
      redis: {
        enabled: true,
        url: process.env.STAGING_REDIS_URL
      },
      monitoring: {
        enabled: true,
        service: 'vercel-analytics'
      }
    },
    production: {
      name: 'Production',
      url: 'https://ai-news-dashboard.com',
      apiUrl: 'https://api.ai-news-dashboard.com',
      features: {
        experimentationEngine: true,
        contextAwareness: true,
        socialFeatures: true,
        voiceSearch: true,
        realTimeCollaboration: true,
        advancedAI: true,
        analytics: true
      },
      database: {
        type: 'postgresql',
        url: process.env.DATABASE_URL,
        ssl: true,
        poolSize: 20
      },
      redis: {
        enabled: true,
        url: process.env.REDIS_URL,
        cluster: true
      },
      monitoring: {
        enabled: true,
        service: 'datadog',
        errorTracking: 'sentry'
      },
      cdn: {
        enabled: true,
        provider: 'cloudflare',
        caching: {
          static: '1y',
          api: '5m',
          dynamic: '1h'
        }
      }
    }
  },

  // Feature Flag Rollout Strategy
  featureRollout: {
    'voice-search': {
      development: 1.0,
      staging: 1.0,
      production: 0.8
    },
    'real-time-collaboration': {
      development: 0.5,
      staging: 0.8,
      production: 0.3
    },
    'advanced-ai-skills': {
      development: 1.0,
      staging: 1.0,
      production: 0.6
    },
    'contextual-notifications': {
      development: 1.0,
      staging: 1.0,
      production: 0.9
    }
  },

  // A/B Test Configuration
  experiments: {
    'context-search-triggers': {
      traffic: 0.8, // 80% of users
      environments: ['staging', 'production']
    },
    'ai-skill-presentation': {
      traffic: 0.6, // 60% of users
      environments: ['staging', 'production']
    },
    'personalization-depth': {
      traffic: 1.0, // 100% of users
      environments: ['development', 'staging', 'production']
    }
  },

  // Performance Targets (OKRs)
  performanceTargets: {
    // Core Web Vitals
    lcp: 2.5, // Largest Contentful Paint (seconds)
    fid: 100, // First Input Delay (milliseconds)
    cls: 0.1, // Cumulative Layout Shift
    
    // Custom Metrics
    searchEngagement: 0.15, // +15% typeahead engagement
    skillUsage: 0.25, // 25% of users use AI skills
    socialEngagement: 0.10, // 10% social feature adoption
    contextAccuracy: 0.85, // 85% context prediction accuracy
    
    // Business Metrics
    dailyActiveUsers: 1000,
    sessionDuration: 300, // 5 minutes average
    returnRate: 0.4, // 40% return within 7 days
    conversionRate: 0.05 // 5% conversion to premium
  },

  // Monitoring & Analytics
  monitoring: {
    metrics: {
      // User Behavior
      'user.session.duration': 'histogram',
      'user.search.queries': 'counter',
      'user.article.reads': 'counter',
      'user.social.interactions': 'counter',
      'user.skill.usage': 'counter',
      
      // Performance
      'app.load.time': 'histogram',
      'api.response.time': 'histogram',
      'search.response.time': 'histogram',
      'ai.processing.time': 'histogram',
      
      // Context Awareness
      'context.prediction.accuracy': 'gauge',
      'context.trigger.effectiveness': 'histogram',
      'personalization.relevance': 'gauge',
      
      // Experiments
      'experiment.conversion': 'counter',
      'experiment.engagement': 'histogram',
      'feature.adoption': 'gauge'
    },
    
    alerts: {
      'high-error-rate': {
        condition: 'error_rate > 0.05',
        channels: ['email', 'slack']
      },
      'slow-response': {
        condition: 'avg_response_time > 2000',
        channels: ['slack']
      },
      'low-engagement': {
        condition: 'daily_active_users < 500',
        channels: ['email']
      }
    }
  },

  // Security Configuration
  security: {
    cors: {
      origins: {
        development: ['http://localhost:3000', 'http://localhost:3001'],
        staging: ['https://staging-ai-news.vercel.app'],
        production: ['https://ai-news-dashboard.com']
      }
    },
    
    rateLimit: {
      api: {
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 100 // requests per window
      },
      search: {
        windowMs: 60 * 1000, // 1 minute
        max: 30 // searches per minute
      },
      ai: {
        windowMs: 60 * 1000, // 1 minute
        max: 10 // AI requests per minute
      }
    },
    
    headers: {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
      'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https:;"
    }
  },

  // Build Configuration
  build: {
    optimization: {
      splitChunks: true,
      treeshaking: true,
      minification: true,
      compression: 'gzip'
    },
    
    bundleAnalysis: {
      enabled: true,
      maxSize: {
        initial: '500kb',
        async: '250kb'
      }
    },
    
    assets: {
      optimization: true,
      formats: ['webp', 'avif'],
      sizes: [320, 640, 960, 1280, 1920]
    }
  },

  // Deployment Strategy
  deployment: {
    strategy: 'blue-green',
    
    steps: [
      'build',
      'test',
      'security-scan',
      'performance-test',
      'deploy-staging',
      'integration-test',
      'deploy-production',
      'health-check',
      'rollback-on-failure'
    ],
    
    rollback: {
      automatic: true,
      triggers: [
        'error_rate > 0.1',
        'response_time > 5000',
        'availability < 0.99'
      ]
    },
    
    healthChecks: {
      '/health': {
        timeout: 5000,
        interval: 30000
      },
      '/api/health': {
        timeout: 3000,
        interval: 15000
      }
    }
  },

  // API Configuration
  api: {
    endpoints: {
      news: {
        provider: 'newsapi',
        fallbacks: ['gnews', 'currents'],
        caching: '5m',
        rateLimit: '100/hour'
      },
      
      ai: {
        provider: 'openai',
        fallbacks: ['anthropic', 'cohere'],
        caching: '1h',
        rateLimit: '50/hour'
      },
      
      search: {
        provider: 'elasticsearch',
        fallbacks: ['algolia'],
        caching: '10m',
        rateLimit: '200/hour'
      }
    },
    
    middleware: [
      'cors',
      'helmet',
      'compression',
      'rate-limit',
      'auth',
      'logging',
      'error-handling'
    ]
  }
};

// Environment-specific overrides
const getConfig = (environment = 'development') => {
  const baseConfig = deploymentConfig;
  const envConfig = baseConfig.environments[environment];
  
  return {
    ...baseConfig,
    environment,
    ...envConfig,
    features: {
      ...baseConfig.environments.development.features,
      ...envConfig.features
    }
  };
};

// Export configuration
module.exports = {
  deploymentConfig,
  getConfig,
  
  // Helper functions
  isFeatureEnabled: (feature, environment = 'development') => {
    const config = getConfig(environment);
    return config.features[feature] || false;
  },
  
  getFeatureRollout: (feature, environment = 'development') => {
    return deploymentConfig.featureRollout[feature]?.[environment] || 0;
  },
  
  shouldRunExperiment: (experiment, environment = 'development') => {
    const expConfig = deploymentConfig.experiments[experiment];
    return expConfig && expConfig.environments.includes(environment);
  }
};

// Development helper
if (process.env.NODE_ENV === 'development') {
  console.log('🚀 AI News Dashboard - Development Configuration Loaded');
  console.log('📊 Experimentation Engine: Active');
  console.log('🧠 Context Awareness: Enabled');
  console.log('🎯 Performance Targets:', deploymentConfig.performanceTargets);
}