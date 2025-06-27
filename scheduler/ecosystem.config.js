// PM2 Ecosystem Configuration for RSE Scheduler Orchestrator
// Production-ready process management with clustering and monitoring

module.exports = {
  apps: [
    {
      // Main RSE Scheduler application
      name: 'rse-scheduler',
      script: 'index.js',
      cwd: __dirname,
      
      // Process management
      instances: 1, // Single instance for scheduler to avoid conflicts
      exec_mode: 'fork', // Fork mode for single instance
      
      // Environment configuration
      env: {
        NODE_ENV: 'development',
        LOG_LEVEL: 'debug',
        PORT: 9090
      },
      env_staging: {
        NODE_ENV: 'staging',
        LOG_LEVEL: 'info',
        PORT: 9090
      },
      env_production: {
        NODE_ENV: 'production',
        LOG_LEVEL: 'info',
        PORT: 9090
      },
      
      // Restart configuration
      autorestart: true,
      watch: false, // Disable watch in production
      max_memory_restart: '1G',
      restart_delay: 4000,
      max_restarts: 10,
      min_uptime: '10s',
      
      // Logging configuration
      log_file: './logs/rse-scheduler.log',
      out_file: './logs/rse-scheduler-out.log',
      error_file: './logs/rse-scheduler-error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      
      // Advanced options
      kill_timeout: 5000,
      listen_timeout: 3000,
      shutdown_with_message: true,
      
      // Health monitoring
      health_check_grace_period: 3000,
      
      // Source map support
      source_map_support: true,
      
      // Instance variables
      instance_var: 'INSTANCE_ID',
      
      // Cron restart (optional - restart daily at 3 AM)
      cron_restart: '0 3 * * *',
      
      // Node.js options
      node_args: [
        '--max-old-space-size=1024',
        '--optimize-for-size'
      ],
      
      // Process title
      name: 'rse-scheduler-main'
    },
    
    {
      // Health check service (optional separate process)
      name: 'rse-scheduler-health',
      script: 'health-check.js',
      cwd: __dirname,
      
      // Process management
      instances: 1,
      exec_mode: 'fork',
      
      // Environment
      env: {
        NODE_ENV: 'production',
        HEALTH_CHECK_INTERVAL: 30000,
        MAIN_SERVICE_URL: 'http://localhost:9090'
      },
      
      // Restart configuration
      autorestart: true,
      watch: false,
      max_memory_restart: '256M',
      restart_delay: 2000,
      
      // Logging
      log_file: './logs/health-check.log',
      out_file: './logs/health-check-out.log',
      error_file: './logs/health-check-error.log',
      
      // Only run in production
      env_filter: ['production']
    },
    
    {
      // Metrics exporter (optional)
      name: 'rse-scheduler-metrics',
      script: 'metrics-exporter.js',
      cwd: __dirname,
      
      // Process management
      instances: 1,
      exec_mode: 'fork',
      
      // Environment
      env: {
        NODE_ENV: 'production',
        METRICS_PORT: 9091,
        EXPORT_INTERVAL: 60000
      },
      
      // Restart configuration
      autorestart: true,
      watch: false,
      max_memory_restart: '256M',
      
      // Logging
      log_file: './logs/metrics.log',
      
      // Only run if metrics are enabled
      env_filter: ['production']
    }
  ],
  
  // Deployment configuration
  deploy: {
    production: {
      user: 'deploy',
      host: ['production-server-1', 'production-server-2'],
      ref: 'origin/main',
      repo: 'git@github.com:your-org/ai-news-dashboard.git',
      path: '/var/www/rse-scheduler',
      'post-deploy': 'npm install && npm run build && pm2 reload ecosystem.config.js --env production',
      'pre-setup': 'apt update && apt install git -y',
      'post-setup': 'ls -la',
      ssh_options: 'StrictHostKeyChecking=no',
      key: '~/.ssh/deploy_key'
    },
    
    staging: {
      user: 'deploy',
      host: 'staging-server',
      ref: 'origin/develop',
      repo: 'git@github.com:your-org/ai-news-dashboard.git',
      path: '/var/www/rse-scheduler-staging',
      'post-deploy': 'npm install && npm run build && pm2 reload ecosystem.config.js --env staging',
      'pre-setup': 'apt update && apt install git -y'
    }
  },
  
  // Global PM2 configuration
  pmx: {
    enabled: true,
    network: true,
    ports: true,
    
    // Custom metrics
    custom_probes: true,
    
    // Transaction tracing
    transactions: true,
    
    // Profiling
    profiling: true
  },
  
  // Keymetrics configuration (optional)
  keymetrics: {
    public_key: process.env.KEYMETRICS_PUBLIC_KEY,
    secret_key: process.env.KEYMETRICS_SECRET_KEY,
    machine_name: process.env.MACHINE_NAME || 'rse-scheduler-server'
  }
};

// Helper functions for dynamic configuration
function getInstanceCount() {
  const cpus = require('os').cpus().length;
  return process.env.NODE_ENV === 'production' ? 1 : 1; // Always 1 for scheduler
}

function getMemoryLimit() {
  const totalMem = require('os').totalmem();
  const memInGB = Math.floor(totalMem / (1024 * 1024 * 1024));
  return memInGB > 4 ? '2G' : '1G';
}

// Export configuration with environment-specific overrides
if (process.env.NODE_ENV === 'development') {
  module.exports.apps[0].watch = true;
  module.exports.apps[0].ignore_watch = ['node_modules', 'logs', '*.log'];
  module.exports.apps[0].watch_options = {
    followSymlinks: false,
    usePolling: true,
    interval: 1000
  };
}

// Production optimizations
if (process.env.NODE_ENV === 'production') {
  module.exports.apps[0].max_memory_restart = getMemoryLimit();
  module.exports.apps[0].node_args.push('--enable-source-maps');
}