# RSE Scheduler Orchestrator

🚀 **A powerful, production-ready job scheduler for the RSE News Dashboard ecosystem**

The RSE Scheduler Orchestrator is a comprehensive cron-like job scheduling system designed specifically for end-to-end RSE (Research Software Engineering) news updates. It provides advanced features like dependency management, retry logic, failure notifications, and comprehensive monitoring.

## ✨ Features

- **🕐 Dynamic Task Scheduling**: Flexible cron-based scheduling with timezone support
- **🔗 Dependency Management**: Define job dependencies with automatic waiting and validation
- **🔄 Retry & Recovery**: Intelligent retry logic with exponential backoff
- **📧 Multi-channel Notifications**: Email, Slack, and webhook notifications for job events
- **📊 Monitoring & Metrics**: Prometheus metrics and health checks
- **🛡️ Resource Management**: Concurrent job limits and timeout handling
- **📝 Comprehensive Logging**: Structured logging with Winston
- **🧪 Test Coverage**: Extensive test suite with Jest

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   jobs.yaml     │───▶│  RSE Scheduler   │───▶│   Job Execution │
│  Configuration  │    │   Orchestrator   │    │    & Monitoring │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Notifications  │
                       │ Email│Slack│Web │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Node.js >= 16.0.0
- npm >= 8.0.0

### Installation

```bash
# Navigate to the scheduler directory
cd scheduler

# Install dependencies
npm install

# Validate configuration
npm run validate-config

# Start the scheduler
npm start
```

### Development Mode

```bash
# Start with auto-reload
npm run dev

# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Check code quality
npm run lint
npm run lint:fix
```

## ⚙️ Configuration

The scheduler is configured via `jobs.yaml`. Here's the structure:

### Basic Job Configuration

```yaml
scheduler:
  timezone: "UTC"
  max_concurrent_jobs: 5
  retry_policy:
    max_retries: 3
    retry_delay: "5m"
    exponential_backoff: true

jobs:
  theme_extraction:
    enabled: true
    schedule: "0 8 * * *"  # Daily at 8 AM UTC
    command:
      type: "node"
      script: "../agents/theme_extractor.js"
      args: ["--mode", "daily"]
    timeout: "30m"
    priority: "high"
    dependencies: []
    notifications:
      on_success:
        - type: "email"
          recipients: ["admin@example.com"]
      on_failure:
        - type: "slack"
          channel: "#alerts"

  news_ingestion:
    enabled: true
    schedule: "0 6 * * *"  # Daily at 6 AM UTC
    command:
      type: "python"
      script: "../scrapers/news_scraper.py"
    timeout: "45m"
    dependencies: []  # No dependencies
    
  daily_summary:
    enabled: true
    schedule: "0 10 * * *"  # Daily at 10 AM UTC
    command:
      type: "node"
      script: "../agents/summarizer.js"
    timeout: "20m"
    dependencies: ["theme_extraction", "news_ingestion"]  # Wait for these jobs
```

### Notification Configuration

```yaml
notifications:
  email:
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    smtp_user: "${SMTP_USER}"  # Environment variable
    smtp_password: "${SMTP_PASSWORD}"
    from_address: "scheduler@yourcompany.com"
    
  slack:
    webhook_url: "${SLACK_WEBHOOK_URL}"
    channel: "#rse-alerts"
    username: "RSE Scheduler"
    
  webhook:
    default_timeout: "10s"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
```

### Monitoring Configuration

```yaml
monitoring:
  metrics:
    port: 9090
  health_check:
    interval: "30s"
    stuck_job_threshold: 1.5  # 1.5x timeout
```

## 📋 Job Types & Commands

### Node.js Jobs

```yaml
command:
  type: "node"
  script: "path/to/script.js"
  args: ["--env", "production", "--verbose"]
```

### Python Jobs

```yaml
command:
  type: "python"
  script: "path/to/script.py"
  args: ["--config", "config.json"]
```

## 🔗 Dependency Management

Jobs can depend on other jobs completing successfully:

```yaml
jobs:
  data_fetch:
    schedule: "0 6 * * *"
    # ... other config
    
  data_process:
    schedule: "0 7 * * *"
    dependencies: ["data_fetch"]  # Waits for data_fetch to complete
    
  report_generation:
    schedule: "0 8 * * *"
    dependencies: ["data_fetch", "data_process"]  # Waits for both
```

**Dependency Features:**
- ✅ Automatic waiting for dependencies
- ✅ Timeout handling (5-minute default)
- ✅ Dependency validation at startup
- ✅ Metrics for dependency wait times

## 🔄 Retry Logic

Robust retry mechanism with configurable policies:

```yaml
scheduler:
  retry_policy:
    max_retries: 3
    retry_delay: "5m"
    exponential_backoff: true  # 5m, 10m, 20m delays

# Or per-job retry policy
jobs:
  critical_job:
    retry_policy:
      max_retries: 5
      retry_delay: "2m"
      exponential_backoff: false  # Fixed 2m delay
```

## 📧 Notifications

### Email Notifications

```yaml
notifications:
  on_success:
    - type: "email"
      recipients: ["team@company.com", "admin@company.com"]
      subject: "Job Completed: {job_name}"
      
  on_failure:
    - type: "email"
      recipients: ["alerts@company.com"]
      subject: "🚨 Job Failed: {job_name}"
```

### Slack Notifications

```yaml
notifications:
  on_failure:
    - type: "slack"
      channel: "#alerts"
      mention_users: ["@oncall", "@admin"]
```

### Webhook Notifications

```yaml
notifications:
  on_success:
    - type: "webhook"
      url: "https://api.company.com/job-webhook"
      headers:
        Authorization: "Bearer ${WEBHOOK_TOKEN}"
```

## 📊 Monitoring & Metrics

### Prometheus Metrics

The scheduler exposes metrics at `http://localhost:9090/metrics`:

- `rse_scheduler_job_executions_total` - Total job executions by status
- `rse_scheduler_job_duration_seconds` - Job execution duration
- `rse_scheduler_active_jobs` - Currently running jobs
- `rse_scheduler_health` - Scheduler health status
- `rse_scheduler_dependency_wait_seconds` - Dependency wait times

### Health Endpoint

Health check available at `http://localhost:9090/health`:

```bash
curl http://localhost:9090/health
```

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "active_jobs": 2,
  "registered_jobs": 8
}
```

### Monitoring Commands

```bash
# Check health
npm run health-check

# View metrics
npm run metrics

# View logs
tail -f logs/scheduler.log
```

## 🛠️ API Usage

### Programmatic Control

```javascript
const RSESchedulerOrchestrator = require('./index');

const scheduler = new RSESchedulerOrchestrator('./jobs.yaml');

// Start the scheduler
scheduler.start();

// Manually trigger a job
await scheduler.triggerJob('theme_extraction');

// Get job status
const status = scheduler.getJobStatus('news_ingestion');
console.log(status);

// Listen to events
scheduler.on('started', () => console.log('Scheduler started'));
scheduler.on('health-check', (status) => console.log('Health:', status));

// Graceful shutdown
process.on('SIGINT', () => {
  scheduler.stop();
  process.exit(0);
});
```

### Job Status Response

```javascript
{
  name: 'theme_extraction',
  config: { /* job configuration */ },
  lastRun: '2024-01-15T08:00:00.000Z',
  runCount: 45,
  failureCount: 2,
  currentState: 'idle', // 'idle', 'running', 'failed'
  history: [
    {
      id: 'theme_extraction-1705305600000',
      state: 'completed',
      startTime: 1705305600000,
      endTime: 1705305800000,
      output: 'Extracted 15 themes successfully'
    }
    // ... last 10 executions
  ]
}
```

## 🔧 Environment Variables

Set these environment variables for production:

```bash
# Email configuration
export SMTP_HOST="smtp.gmail.com"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"

# Slack configuration
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# API tokens
export API_TOKEN="your-api-token"
export WEBHOOK_TOKEN="your-webhook-token"

# Node environment
export NODE_ENV="production"
```

## 📁 Directory Structure

```
scheduler/
├── index.js              # Main scheduler orchestrator
├── package.json          # Dependencies and scripts
├── jobs.yaml            # Job configuration
├── scheduler.test.js    # Test suite
├── README.md           # This documentation
├── logs/               # Log files (auto-created)
│   ├── scheduler.log
│   └── scheduler-error.log
└── node_modules/       # Dependencies
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run specific test file
npm test scheduler.test.js

# Run tests in watch mode
npm run test:watch
```

### Test Coverage

The test suite covers:
- ✅ Job registration and scheduling
- ✅ Dependency management
- ✅ Retry logic and failure handling
- ✅ Notification systems
- ✅ Health monitoring
- ✅ Configuration validation
- ✅ Graceful shutdown

## 🚀 Production Deployment

### Using PM2

```bash
# Install PM2 globally
npm install -g pm2

# Start with PM2
pm2 start index.js --name "rse-scheduler"

# Monitor
pm2 monit

# View logs
pm2 logs rse-scheduler
```

### Using Docker

```dockerfile
FROM node:16-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 9090
CMD ["npm", "start"]
```

### Using systemd

```ini
# /etc/systemd/system/rse-scheduler.service
[Unit]
Description=RSE Scheduler Orchestrator
After=network.target

[Service]
Type=simple
User=rse
WorkingDirectory=/opt/rse-scheduler
ExecStart=/usr/bin/node index.js
Restart=always
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
```

## 🔍 Troubleshooting

### Common Issues

**Jobs not starting:**
```bash
# Check configuration
npm run validate-config

# Check logs
tail -f logs/scheduler-error.log
```

**Dependency timeouts:**
```yaml
# Increase timeout in jobs.yaml
scheduler:
  dependency_timeout: "10m"  # Default is 5m
```

**Memory issues:**
```bash
# Monitor memory usage
node --max-old-space-size=4096 index.js
```

**Stuck jobs:**
```bash
# Check health endpoint
curl http://localhost:9090/health

# View metrics
curl http://localhost:9090/metrics | grep stuck
```

### Debug Mode

```bash
# Enable debug logging
DEBUG=rse-scheduler:* npm start

# Or set log level
LOG_LEVEL=debug npm start
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests: `npm test`
5. Lint your code: `npm run lint:fix`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Follow the existing code style
- Update documentation
- Add appropriate logging
- Handle errors gracefully

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For support and questions:

- 📧 Email: rse-team@yourcompany.com
- 💬 Slack: #rse-support
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/ai-news-dashboard/issues)
- 📖 Wiki: [Project Wiki](https://github.com/your-org/ai-news-dashboard/wiki)

---

**Built with ❤️ by the RSE Team**