# Scheduler Module Documentation

## Overview

Dr. NewsForge's Advanced Scheduler Module provides automated job orchestration for the AI News Dashboard. This YAML-based scheduling system manages periodic tasks including news ingestion, theme extraction, report generation, and system maintenance with enterprise-grade reliability and monitoring.

## Features

### 🚀 Core Capabilities
- **YAML-based Configuration**: Human-readable job definitions
- **Cron Scheduling**: Flexible timing with standard cron expressions
- **Dependency Management**: Job execution ordering and prerequisites
- **Resource Management**: CPU, memory, and concurrency limits
- **Retry Logic**: Automatic failure recovery with exponential backoff
- **Health Monitoring**: Built-in health checks and alerting

### 📊 Job Types
- **Data Ingestion**: Automated news fetching and processing
- **Analytics**: Theme extraction and trend analysis
- **Reporting**: Daily, weekly, and monthly report generation
- **Maintenance**: Cache cleanup, log rotation, and backups
- **Monitoring**: Health checks and performance metrics
- **Notifications**: Alert delivery and status updates

### ⚡ Enterprise Features
- **High Availability**: Multi-instance coordination
- **Load Balancing**: Distributed job execution
- **Audit Logging**: Comprehensive execution tracking
- **Security**: Role-based access and encrypted communications
- **Scalability**: Horizontal scaling support
- **Integration**: Webhook and API connectivity

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Job Configs   │────│   Scheduler      │────│   Executors     │
│                 │    │     Engine       │    │                 │
│ • YAML Files    │    │                  │    │ • Node.js       │
│ • Cron Schedules│    │ • Job Queue      │    │ • Python        │
│ • Dependencies  │    │ • State Manager  │    │ • Shell Scripts │
│ • Resources     │    │ • Health Monitor │    │ • Docker        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │   Monitoring     │
                       │                  │
                       │ • Metrics        │
                       │ • Alerts         │
                       │ • Logs           │
                       │ • Dashboards     │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │   Integrations   │
                       │                  │
                       │ • Webhooks       │
                       │ • Notifications  │
                       │ • External APIs  │
                       │ • Backup Systems │
                       └──────────────────┘
```

## Job Configuration

### Basic Job Structure

```yaml
jobs:
  job_name:
    name: "Human Readable Name"
    description: "Job description"
    schedule: "0 * * * *"  # Cron expression
    enabled: true
    priority: "high"  # low, medium, high, critical
    timeout: "15m"
    retries: 3
    
    command:
      type: "node"  # node, python, shell, docker
      script: "path/to/script.js"
      args:
        - "--option=value"
        - "--flag"
    
    resources:
      memory: "512MB"
      cpu: "0.5"
      disk: "1GB"
    
    dependencies:
      - "prerequisite_job"
    
    notifications:
      on_success: ["webhook", "email"]
      on_failure: ["slack", "email", "webhook"]
```

### Complete Job Example

```yaml
jobs:
  theme_extraction:
    name: "Hourly Top Themes Extraction"
    description: "Analyze latest articles to extract trending themes and sentiment"
    schedule: "0 * * * *"  # Every hour at minute 0
    enabled: true
    priority: "high"
    timeout: "15m"
    retries: 3
    retry_delay: "2m"
    
    command:
      type: "node"
      script: "analytics/themes.js"
      args:
        - "--mode=scheduled"
        - "--max-articles=100"
        - "--output-dir=analytics/output"
    
    environment:
      NODE_ENV: "production"
      ANALYTICS_MAX_ARTICLES: "100"
      ANALYTICS_MIN_THEME_SCORE: "0.3"
    
    resources:
      memory: "1GB"
      cpu: "1.0"
      disk: "500MB"
      max_concurrent: 1
    
    dependencies:
      - "news_ingestion"
    
    health_check:
      endpoint: "/api/analytics/health"
      interval: "5m"
      timeout: "30s"
      retries: 3
    
    notifications:
      on_success: ["webhook"]
      on_failure: ["slack", "email", "webhook"]
      on_timeout: ["slack", "webhook"]
    
    monitoring:
      metrics:
        - "themes_extracted_total"
        - "processing_duration_seconds"
        - "memory_usage_bytes"
      alerts:
        - name: "Low Theme Count"
          condition: "themes_extracted_total < 5"
          severity: "warning"
```

## Predefined Jobs

### News Ingestion

```yaml
news_ingestion:
  name: "News Articles Ingestion"
  description: "Fetch and process news from RSS feeds and APIs"
  schedule: "*/30 * * * *"  # Every 30 minutes
  enabled: true
  priority: "high"
  timeout: "10m"
  
  command:
    type: "node"
    script: "news/ingest.js"
    args:
      - "--sources=all"
      - "--max-articles=200"
  
  resources:
    memory: "512MB"
    cpu: "0.5"
```

### Theme Extraction

```yaml
theme_extraction:
  name: "Hourly Top Themes Extraction"
  schedule: "0 * * * *"  # Every hour
  dependencies: ["news_ingestion"]
  
  command:
    type: "node"
    script: "analytics/themes.js"
    args: ["--mode=scheduled"]
```

### Daily Reports

```yaml
daily_report:
  name: "Daily Analytics Report"
  schedule: "0 6 * * *"  # 6 AM daily
  dependencies: ["theme_extraction"]
  
  command:
    type: "node"
    script: "reports/daily.js"
    args: ["--format=html", "--email=true"]
```

### System Maintenance

```yaml
cache_cleanup:
  name: "Cache Cleanup"
  schedule: "0 2 * * *"  # 2 AM daily
  
  command:
    type: "shell"
    script: "scripts/cleanup.sh"
    args: ["--cache-dir=./cache", "--max-age=24h"]
```

## Scheduler Configuration

### Global Settings

```yaml
scheduler:
  timezone: "UTC"
  max_concurrent_jobs: 10
  job_history_retention: "30d"
  log_level: "info"
  
  database:
    type: "sqlite"
    path: "./scheduler/jobs.db"
    backup_interval: "6h"
  
  api:
    enabled: true
    port: 8080
    auth:
      type: "api_key"
      key: "${SCHEDULER_API_KEY}"
  
  monitoring:
    enabled: true
    metrics_port: 9090
    health_check_interval: "1m"
    
  notifications:
    default_channels: ["webhook"]
    rate_limit: "10/hour"
```

### Environment Overrides

```yaml
environments:
  development:
    scheduler:
      log_level: "debug"
      max_concurrent_jobs: 3
    jobs:
      theme_extraction:
        schedule: "*/5 * * * *"  # Every 5 minutes
        
  production:
    scheduler:
      max_concurrent_jobs: 20
      database:
        type: "postgresql"
        url: "${DATABASE_URL}"
    jobs:
      theme_extraction:
        resources:
          memory: "2GB"
          cpu: "2.0"
```

## Notification Channels

### Webhook Configuration

```yaml
notifications:
  webhook:
    url: "https://api.newsforge.ai/webhooks/scheduler"
    method: "POST"
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
      Content-Type: "application/json"
    timeout: "10s"
    retries: 3
    
    payload_template: |
      {
        "job": "{{ job.name }}",
        "status": "{{ job.status }}",
        "timestamp": "{{ job.timestamp }}",
        "duration": "{{ job.duration }}",
        "message": "{{ job.message }}"
      }
```

### Email Configuration

```yaml
notifications:
  email:
    smtp:
      host: "smtp.gmail.com"
      port: 587
      secure: false
      auth:
        user: "${SMTP_USER}"
        pass: "${SMTP_PASSWORD}"
    
    from: "scheduler@newsforge.ai"
    to: ["admin@newsforge.ai", "alerts@newsforge.ai"]
    
    templates:
      success: "Job {{ job.name }} completed successfully"
      failure: "Job {{ job.name }} failed: {{ job.error }}"
      timeout: "Job {{ job.name }} timed out after {{ job.timeout }}"
```

### Slack Configuration

```yaml
notifications:
  slack:
    webhook_url: "${SLACK_WEBHOOK_URL}"
    channel: "#alerts"
    username: "Dr. NewsForge Scheduler"
    icon_emoji: ":robot_face:"
    
    message_template: |
      :{{ status_emoji }}: *{{ job.name }}* {{ job.status }}
      Duration: {{ job.duration }}
      {% if job.error %}Error: {{ job.error }}{% endif %}
```

## API Reference

### Job Management

#### List Jobs
```http
GET /api/jobs
Authorization: Bearer <api_key>
```

**Response:**
```json
{
  "jobs": [
    {
      "id": "theme_extraction",
      "name": "Hourly Top Themes Extraction",
      "status": "running",
      "next_run": "2024-01-15T11:00:00Z",
      "last_run": "2024-01-15T10:00:00Z",
      "success_rate": 98.5
    }
  ]
}
```

#### Get Job Details
```http
GET /api/jobs/{job_id}
Authorization: Bearer <api_key>
```

**Response:**
```json
{
  "id": "theme_extraction",
  "name": "Hourly Top Themes Extraction",
  "description": "Analyze latest articles to extract trending themes",
  "schedule": "0 * * * *",
  "enabled": true,
  "status": "idle",
  "next_run": "2024-01-15T11:00:00Z",
  "last_run": {
    "timestamp": "2024-01-15T10:00:00Z",
    "duration": "2m 15s",
    "status": "success",
    "output": "Extracted 15 themes from 100 articles"
  },
  "statistics": {
    "total_runs": 1247,
    "successful_runs": 1228,
    "failed_runs": 19,
    "success_rate": 98.5,
    "average_duration": "2m 8s"
  }
}
```

#### Trigger Job
```http
POST /api/jobs/{job_id}/trigger
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "args": ["--max-articles=50"],
  "priority": "high"
}
```

#### Enable/Disable Job
```http
PATCH /api/jobs/{job_id}
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "enabled": false
}
```

### Execution History

#### Get Job History
```http
GET /api/jobs/{job_id}/history?limit=10&status=failed
Authorization: Bearer <api_key>
```

**Response:**
```json
{
  "executions": [
    {
      "id": "exec_123456",
      "timestamp": "2024-01-15T10:00:00Z",
      "duration": "2m 15s",
      "status": "success",
      "exit_code": 0,
      "output": "Extracted 15 themes from 100 articles",
      "resources_used": {
        "memory_peak": "456MB",
        "cpu_avg": "0.7"
      }
    }
  ],
  "pagination": {
    "total": 1247,
    "page": 1,
    "limit": 10
  }
}
```

### System Status

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "scheduler": {
    "status": "running",
    "uptime": "5d 12h 30m",
    "active_jobs": 2,
    "queued_jobs": 0
  },
  "database": {
    "status": "connected",
    "size": "45MB",
    "last_backup": "2024-01-15T04:00:00Z"
  },
  "resources": {
    "memory_usage": "512MB",
    "cpu_usage": "15%",
    "disk_usage": "2.3GB"
  }
}
```

## Monitoring and Metrics

### Key Metrics

#### Job Metrics
- `scheduler_jobs_total`: Total number of configured jobs
- `scheduler_jobs_enabled`: Number of enabled jobs
- `scheduler_executions_total`: Total job executions
- `scheduler_execution_duration_seconds`: Job execution time histogram
- `scheduler_execution_failures_total`: Failed job executions
- `scheduler_queue_size`: Current job queue size

#### System Metrics
- `scheduler_uptime_seconds`: Scheduler uptime
- `scheduler_memory_usage_bytes`: Memory consumption
- `scheduler_cpu_usage_percent`: CPU utilization
- `scheduler_active_jobs`: Currently running jobs

#### Performance Metrics
- `scheduler_job_success_rate`: Success rate by job
- `scheduler_average_execution_time`: Average execution duration
- `scheduler_queue_wait_time`: Time jobs spend in queue
- `scheduler_resource_utilization`: Resource usage efficiency

### Prometheus Integration

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"
    
    custom_metrics:
      - name: "themes_extracted_total"
        type: "counter"
        help: "Total themes extracted"
        labels: ["job", "category"]
      
      - name: "articles_processed_total"
        type: "counter"
        help: "Total articles processed"
        labels: ["job", "source"]
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Dr. NewsForge Scheduler",
    "panels": [
      {
        "title": "Job Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(scheduler_executions_total{status=\"success\"}[1h]) / rate(scheduler_executions_total[1h]) * 100"
          }
        ]
      },
      {
        "title": "Execution Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, scheduler_execution_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

## Security

### Authentication

```yaml
security:
  authentication:
    type: "api_key"  # api_key, jwt, oauth2
    api_key:
      header: "Authorization"
      prefix: "Bearer "
      keys:
        - name: "admin"
          key: "${ADMIN_API_KEY}"
          permissions: ["read", "write", "admin"]
        - name: "readonly"
          key: "${READONLY_API_KEY}"
          permissions: ["read"]
```

### Authorization

```yaml
security:
  authorization:
    roles:
      admin:
        permissions:
          - "jobs:*"
          - "system:*"
          - "config:*"
      
      operator:
        permissions:
          - "jobs:read"
          - "jobs:trigger"
          - "jobs:enable"
          - "jobs:disable"
      
      viewer:
        permissions:
          - "jobs:read"
          - "system:health"
```

### Encryption

```yaml
security:
  encryption:
    at_rest:
      enabled: true
      algorithm: "AES-256-GCM"
      key_source: "env"  # env, file, kms
    
    in_transit:
      tls:
        enabled: true
        cert_file: "/etc/ssl/certs/scheduler.crt"
        key_file: "/etc/ssl/private/scheduler.key"
```

## Backup and Recovery

### Backup Configuration

```yaml
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention: "30d"
  
  targets:
    - type: "database"
      path: "./scheduler/jobs.db"
      destination: "s3://newsforge-backups/scheduler/"
    
    - type: "config"
      path: "./scheduler/jobs.yaml"
      destination: "s3://newsforge-backups/config/"
    
    - type: "logs"
      path: "./logs/"
      destination: "s3://newsforge-backups/logs/"
  
  encryption:
    enabled: true
    key: "${BACKUP_ENCRYPTION_KEY}"
  
  notifications:
    on_success: ["email"]
    on_failure: ["slack", "email"]
```

### Disaster Recovery

```yaml
disaster_recovery:
  enabled: true
  
  failover:
    mode: "automatic"  # automatic, manual
    health_check_interval: "30s"
    failure_threshold: 3
    
    secondary_instances:
      - host: "scheduler-backup-1.newsforge.ai"
        priority: 1
      - host: "scheduler-backup-2.newsforge.ai"
        priority: 2
  
  data_replication:
    enabled: true
    mode: "synchronous"  # synchronous, asynchronous
    targets:
      - "scheduler-backup-1.newsforge.ai"
      - "scheduler-backup-2.newsforge.ai"
```

## Deployment

### Docker Deployment

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 8080 9090

CMD ["npm", "run", "scheduler"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  scheduler:
    build: .
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - NODE_ENV=production
      - SCHEDULER_API_KEY=${SCHEDULER_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./scheduler/jobs.yaml:/app/scheduler/jobs.yaml:ro
      - ./logs:/app/logs
    restart: unless-stopped
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scheduler
  labels:
    app: scheduler
spec:
  replicas: 2
  selector:
    matchLabels:
      app: scheduler
  template:
    metadata:
      labels:
        app: scheduler
    spec:
      containers:
      - name: scheduler
        image: newsforge/scheduler:latest
        ports:
        - containerPort: 8080
        - containerPort: 9090
        env:
        - name: NODE_ENV
          value: "production"
        - name: SCHEDULER_API_KEY
          valueFrom:
            secretKeyRef:
              name: scheduler-secrets
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Testing

### Unit Tests

```bash
# Run scheduler tests
npm run test:scheduler

# Test job configuration validation
npm run test:config

# Test notification systems
npm run test:notifications
```

### Integration Tests

```bash
# Test full job execution pipeline
npm run test:integration

# Test API endpoints
npm run test:api

# Test monitoring and metrics
npm run test:monitoring
```

### Load Testing

```bash
# Simulate high job load
npm run load-test:jobs

# Test concurrent execution limits
npm run load-test:concurrency

# Test resource constraints
npm run load-test:resources
```

## Troubleshooting

### Common Issues

#### Jobs Not Running
```bash
# Check scheduler status
curl http://localhost:8080/api/health

# Verify job configuration
node -e "console.log(require('js-yaml').load(require('fs').readFileSync('scheduler/jobs.yaml', 'utf8')))"

# Check job dependencies
curl http://localhost:8080/api/jobs/theme_extraction
```

#### High Resource Usage
```bash
# Monitor resource consumption
top -p $(pgrep -f "scheduler")

# Check concurrent job limits
curl http://localhost:8080/api/jobs | jq '.jobs[] | select(.status == "running")'

# Adjust resource limits
export SCHEDULER_MAX_CONCURRENT_JOBS=5
```

#### Failed Notifications
```bash
# Test webhook connectivity
curl -X POST https://api.newsforge.ai/webhooks/scheduler \
  -H "Authorization: Bearer $WEBHOOK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"test": true}'

# Check email configuration
node -e "require('./lib/notifications').testEmail()"

# Verify Slack webhook
curl -X POST $SLACK_WEBHOOK_URL -d '{"text": "Test message"}'
```

#### Database Issues
```bash
# Check database connectivity
sqlite3 scheduler/jobs.db ".tables"

# Verify database integrity
sqlite3 scheduler/jobs.db "PRAGMA integrity_check;"

# Backup and restore
cp scheduler/jobs.db scheduler/jobs.db.backup
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=scheduler:*

# Run with verbose output
node scheduler/index.js --debug --verbose

# Save debug information
node scheduler/index.js --debug --log-file=debug.log
```

## Performance Optimization

### Job Optimization

1. **Batch Processing**
   - Group related operations
   - Minimize I/O operations
   - Use efficient data structures

2. **Resource Management**
   - Set appropriate memory limits
   - Use CPU limits for fair sharing
   - Implement disk space monitoring

3. **Scheduling Optimization**
   - Distribute jobs across time slots
   - Avoid peak resource periods
   - Use job priorities effectively

### System Tuning

```yaml
performance:
  job_queue:
    max_size: 1000
    batch_size: 10
    processing_interval: "1s"
  
  database:
    connection_pool_size: 10
    query_timeout: "30s"
    vacuum_interval: "24h"
  
  monitoring:
    metrics_buffer_size: 1000
    metrics_flush_interval: "10s"
```

## Roadmap

### Phase 1: Core Features ✅
- [x] YAML-based job configuration
- [x] Cron scheduling
- [x] Basic notifications
- [x] REST API
- [x] Health monitoring

### Phase 2: Enterprise Features 🚧
- [ ] High availability setup
- [ ] Advanced authentication
- [ ] Distributed execution
- [ ] Enhanced monitoring
- [ ] Backup automation

### Phase 3: Advanced Features 📋
- [ ] Machine learning-based scheduling
- [ ] Predictive failure detection
- [ ] Auto-scaling capabilities
- [ ] Advanced workflow orchestration
- [ ] Multi-cloud deployment

### Phase 4: AI Integration 🔮
- [ ] Intelligent job optimization
- [ ] Automated performance tuning
- [ ] Predictive maintenance
- [ ] Smart resource allocation
- [ ] Anomaly detection

## Contributing

### Development Setup

```bash
# Install dependencies
npm install

# Set up environment
cp .env.example .env.local

# Run tests
npm run test:scheduler

# Start development mode
npm run dev:scheduler
```

### Code Guidelines

- Follow ESLint configuration
- Write comprehensive tests
- Document all configuration options
- Use TypeScript for new features
- Follow semantic versioning

### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit PR with detailed description
6. Address review feedback
7. Merge after approval

---

**Dr. NewsForge's Scheduler Module** - Orchestrating intelligent automation for the AI News Dashboard with enterprise-grade reliability and monitoring.