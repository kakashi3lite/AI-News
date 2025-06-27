/**
 * RSE Scheduler Orchestrator v1.0
 * Configures and oversees cron-like jobs for end-to-end RSE news updates
 * 
 * Features:
 * - Dynamic task scheduling with cron expressions
 * - Dependency management between fetch and summary jobs
 * - Retry and failure notifications
 * - Integration with monitoring dashboards
 * - Health checks and metrics collection
 */

const cron = require('node-cron');
const yaml = require('js-yaml');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const EventEmitter = require('events');
const axios = require('axios');
const nodemailer = require('nodemailer');
const winston = require('winston');
const prometheus = require('prom-client');

class RSESchedulerOrchestrator extends EventEmitter {
  constructor(configPath = './jobs.yaml') {
    super();
    this.configPath = configPath;
    this.config = null;
    this.jobs = new Map();
    this.runningJobs = new Map();
    this.jobHistory = new Map();
    this.metrics = this.initializeMetrics();
    this.logger = this.initializeLogger();
    this.emailTransporter = null;
    
    // Job execution states
    this.JOB_STATES = {
      PENDING: 'pending',
      RUNNING: 'running',
      COMPLETED: 'completed',
      FAILED: 'failed',
      RETRYING: 'retrying',
      CANCELLED: 'cancelled'
    };
    
    this.loadConfiguration();
    this.initializeNotifications();
    this.startHealthCheck();
  }

  /**
   * Initialize Prometheus metrics for monitoring
   */
  initializeMetrics() {
    const register = new prometheus.Registry();
    
    const metrics = {
      jobExecutions: new prometheus.Counter({
        name: 'rse_scheduler_job_executions_total',
        help: 'Total number of job executions',
        labelNames: ['job_name', 'status'],
        registers: [register]
      }),
      
      jobDuration: new prometheus.Histogram({
        name: 'rse_scheduler_job_duration_seconds',
        help: 'Job execution duration in seconds',
        labelNames: ['job_name'],
        buckets: [1, 5, 10, 30, 60, 300, 600, 1800, 3600],
        registers: [register]
      }),
      
      activeJobs: new prometheus.Gauge({
        name: 'rse_scheduler_active_jobs',
        help: 'Number of currently running jobs',
        registers: [register]
      }),
      
      schedulerHealth: new prometheus.Gauge({
        name: 'rse_scheduler_health',
        help: 'Scheduler health status (1 = healthy, 0 = unhealthy)',
        registers: [register]
      }),
      
      dependencyWaitTime: new prometheus.Histogram({
        name: 'rse_scheduler_dependency_wait_seconds',
        help: 'Time spent waiting for dependencies',
        labelNames: ['job_name', 'dependency'],
        registers: [register]
      })
    };
    
    // Set initial health to healthy
    metrics.schedulerHealth.set(1);
    
    return { ...metrics, register };
  }

  /**
   * Initialize Winston logger
   */
  initializeLogger() {
    return winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
      ),
      defaultMeta: { service: 'rse-scheduler' },
      transports: [
        new winston.transports.File({ 
          filename: './logs/scheduler-error.log', 
          level: 'error',
          maxsize: 100 * 1024 * 1024, // 100MB
          maxFiles: 10
        }),
        new winston.transports.File({ 
          filename: './logs/scheduler.log',
          maxsize: 100 * 1024 * 1024, // 100MB
          maxFiles: 10
        }),
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
          )
        })
      ]
    });
  }

  /**
   * Load and parse the YAML configuration
   */
  loadConfiguration() {
    try {
      const configFile = fs.readFileSync(this.configPath, 'utf8');
      this.config = yaml.load(configFile);
      this.logger.info('Configuration loaded successfully', { 
        jobCount: Object.keys(this.config.jobs || {}).length 
      });
    } catch (error) {
      this.logger.error('Failed to load configuration', { error: error.message });
      throw new Error(`Configuration loading failed: ${error.message}`);
    }
  }

  /**
   * Initialize notification systems (email, webhooks, Slack)
   */
  initializeNotifications() {
    // Initialize email transporter
    if (this.config.notifications?.email) {
      const emailConfig = this.config.notifications.email;
      this.emailTransporter = nodemailer.createTransporter({
        host: process.env.SMTP_HOST || emailConfig.smtp_host,
        port: emailConfig.smtp_port || 587,
        secure: false,
        auth: {
          user: process.env.SMTP_USER || emailConfig.smtp_user,
          pass: process.env.SMTP_PASSWORD || emailConfig.smtp_password
        }
      });
    }
  }

  /**
   * Start the scheduler and register all jobs
   */
  start() {
    this.logger.info('Starting RSE Scheduler Orchestrator');
    
    // Ensure logs directory exists
    if (!fs.existsSync('./logs')) {
      fs.mkdirSync('./logs', { recursive: true });
    }
    
    // Register all jobs from configuration
    for (const [jobName, jobConfig] of Object.entries(this.config.jobs || {})) {
      if (jobConfig.enabled !== false) {
        this.registerJob(jobName, jobConfig);
      } else {
        this.logger.info(`Job ${jobName} is disabled, skipping registration`);
      }
    }
    
    // Start metrics server
    this.startMetricsServer();
    
    this.logger.info('RSE Scheduler Orchestrator started successfully', {
      registeredJobs: this.jobs.size
    });
    
    this.emit('started');
  }

  /**
   * Register a single job with cron scheduler
   */
  registerJob(jobName, jobConfig) {
    try {
      const task = cron.schedule(jobConfig.schedule, async () => {
        await this.executeJob(jobName, jobConfig);
      }, {
        scheduled: false,
        timezone: this.config.scheduler?.timezone || 'UTC'
      });
      
      this.jobs.set(jobName, {
        task,
        config: jobConfig,
        lastRun: null,
        nextRun: null,
        runCount: 0,
        failureCount: 0
      });
      
      task.start();
      
      this.logger.info(`Job registered: ${jobName}`, {
        schedule: jobConfig.schedule,
        priority: jobConfig.priority,
        timeout: jobConfig.timeout
      });
      
    } catch (error) {
      this.logger.error(`Failed to register job: ${jobName}`, { error: error.message });
      throw error;
    }
  }

  /**
   * Execute a job with dependency checking, retry logic, and monitoring
   */
  async executeJob(jobName, jobConfig) {
    const startTime = Date.now();
    const jobId = `${jobName}-${Date.now()}`;
    
    this.logger.info(`Starting job execution: ${jobName}`, { jobId });
    
    // Check if job is already running
    if (this.runningJobs.has(jobName)) {
      this.logger.warn(`Job ${jobName} is already running, skipping execution`);
      return;
    }
    
    // Check concurrent job limit
    if (this.runningJobs.size >= (this.config.scheduler?.max_concurrent_jobs || 5)) {
      this.logger.warn(`Maximum concurrent jobs reached, queuing ${jobName}`);
      setTimeout(() => this.executeJob(jobName, jobConfig), 30000); // Retry in 30 seconds
      return;
    }
    
    const jobState = {
      id: jobId,
      name: jobName,
      state: this.JOB_STATES.PENDING,
      startTime,
      endTime: null,
      retryCount: 0,
      output: '',
      error: null
    };
    
    this.runningJobs.set(jobName, jobState);
    this.metrics.activeJobs.set(this.runningJobs.size);
    
    try {
      // Check dependencies
      if (jobConfig.dependencies && jobConfig.dependencies.length > 0) {
        await this.waitForDependencies(jobName, jobConfig.dependencies);
      }
      
      // Update job state to running
      jobState.state = this.JOB_STATES.RUNNING;
      
      // Execute the job
      const result = await this.runJobCommand(jobName, jobConfig, jobState);
      
      // Job completed successfully
      jobState.state = this.JOB_STATES.COMPLETED;
      jobState.endTime = Date.now();
      
      const duration = (jobState.endTime - jobState.startTime) / 1000;
      
      // Update metrics
      this.metrics.jobExecutions.inc({ job_name: jobName, status: 'success' });
      this.metrics.jobDuration.observe({ job_name: jobName }, duration);
      
      // Update job info
      const jobInfo = this.jobs.get(jobName);
      if (jobInfo) {
        jobInfo.lastRun = new Date();
        jobInfo.runCount++;
      }
      
      this.logger.info(`Job completed successfully: ${jobName}`, {
        jobId,
        duration: `${duration}s`,
        output: result.output?.substring(0, 500) // Truncate long output
      });
      
      // Send success notifications
      await this.sendNotifications(jobName, jobConfig, 'success', {
        duration,
        output: result.output
      });
      
    } catch (error) {
      await this.handleJobFailure(jobName, jobConfig, jobState, error);
    } finally {
      // Clean up
      this.runningJobs.delete(jobName);
      this.metrics.activeJobs.set(this.runningJobs.size);
      
      // Store job history
      if (!this.jobHistory.has(jobName)) {
        this.jobHistory.set(jobName, []);
      }
      const history = this.jobHistory.get(jobName);
      history.push({ ...jobState });
      
      // Keep only last 100 executions
      if (history.length > 100) {
        history.splice(0, history.length - 100);
      }
    }
  }

  /**
   * Wait for job dependencies to complete
   */
  async waitForDependencies(jobName, dependencies) {
    const dependencyStartTime = Date.now();
    
    for (const dependency of dependencies) {
      const dependencyWaitStart = Date.now();
      
      this.logger.info(`Waiting for dependency: ${dependency}`, { job: jobName });
      
      // Check if dependency job exists
      if (!this.jobs.has(dependency)) {
        throw new Error(`Dependency job '${dependency}' not found for job '${jobName}'`);
      }
      
      // Wait for dependency to complete (with timeout)
      const timeout = 300000; // 5 minutes timeout
      const startWait = Date.now();
      
      while (this.runningJobs.has(dependency)) {
        if (Date.now() - startWait > timeout) {
          throw new Error(`Dependency '${dependency}' timeout for job '${jobName}'`);
        }
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
      }
      
      const dependencyWaitTime = (Date.now() - dependencyWaitStart) / 1000;
      this.metrics.dependencyWaitTime.observe(
        { job_name: jobName, dependency },
        dependencyWaitTime
      );
    }
    
    const totalWaitTime = (Date.now() - dependencyStartTime) / 1000;
    this.logger.info(`Dependencies satisfied for ${jobName}`, {
      waitTime: `${totalWaitTime}s`,
      dependencies
    });
  }

  /**
   * Execute the actual job command
   */
  async runJobCommand(jobName, jobConfig, jobState) {
    return new Promise((resolve, reject) => {
      const { command } = jobConfig;
      const timeout = this.parseTimeout(jobConfig.timeout || '10m');
      
      let process;
      
      if (command.type === 'node') {
        process = spawn('node', [command.script, ...(command.args || [])], {
          cwd: process.cwd(),
          env: { ...process.env, JOB_NAME: jobName, JOB_ID: jobState.id }
        });
      } else if (command.type === 'python') {
        process = spawn('python', [command.script, ...(command.args || [])], {
          cwd: process.cwd(),
          env: { ...process.env, JOB_NAME: jobName, JOB_ID: jobState.id }
        });
      } else {
        reject(new Error(`Unsupported command type: ${command.type}`));
        return;
      }
      
      let output = '';
      let errorOutput = '';
      
      process.stdout.on('data', (data) => {
        const chunk = data.toString();
        output += chunk;
        jobState.output += chunk;
      });
      
      process.stderr.on('data', (data) => {
        const chunk = data.toString();
        errorOutput += chunk;
        jobState.output += chunk;
      });
      
      process.on('close', (code) => {
        if (code === 0) {
          resolve({ output, exitCode: code });
        } else {
          reject(new Error(`Job exited with code ${code}. Error: ${errorOutput}`));
        }
      });
      
      process.on('error', (error) => {
        reject(new Error(`Failed to start job process: ${error.message}`));
      });
      
      // Set timeout
      const timeoutId = setTimeout(() => {
        process.kill('SIGTERM');
        reject(new Error(`Job timeout after ${jobConfig.timeout}`));
      }, timeout);
      
      process.on('close', () => {
        clearTimeout(timeoutId);
      });
    });
  }

  /**
   * Handle job failure with retry logic
   */
  async handleJobFailure(jobName, jobConfig, jobState, error) {
    jobState.state = this.JOB_STATES.FAILED;
    jobState.error = error.message;
    jobState.endTime = Date.now();
    
    const duration = (jobState.endTime - jobState.startTime) / 1000;
    
    // Update metrics
    this.metrics.jobExecutions.inc({ job_name: jobName, status: 'failure' });
    
    // Update job info
    const jobInfo = this.jobs.get(jobName);
    if (jobInfo) {
      jobInfo.failureCount++;
    }
    
    this.logger.error(`Job failed: ${jobName}`, {
      jobId: jobState.id,
      duration: `${duration}s`,
      error: error.message,
      retryCount: jobState.retryCount
    });
    
    // Check if we should retry
    const retryPolicy = jobConfig.retry_policy || this.config.scheduler?.retry_policy;
    const maxRetries = retryPolicy?.max_retries || 3;
    
    if (jobState.retryCount < maxRetries) {
      jobState.retryCount++;
      jobState.state = this.JOB_STATES.RETRYING;
      
      const retryDelay = this.parseTimeout(retryPolicy?.retry_delay || '5m');
      const backoffMultiplier = retryPolicy?.exponential_backoff ? Math.pow(2, jobState.retryCount - 1) : 1;
      const actualDelay = retryDelay * backoffMultiplier;
      
      this.logger.info(`Scheduling retry for ${jobName}`, {
        retryCount: jobState.retryCount,
        maxRetries,
        delay: `${actualDelay / 1000}s`
      });
      
      setTimeout(() => {
        this.runningJobs.delete(jobName); // Remove from running jobs to allow retry
        this.executeJob(jobName, jobConfig);
      }, actualDelay);
      
      return;
    }
    
    // Max retries reached, send failure notifications
    await this.sendNotifications(jobName, jobConfig, 'failure', {
      error: error.message,
      retryCount: jobState.retryCount,
      duration
    });
  }

  /**
   * Send notifications (email, webhook, Slack)
   */
  async sendNotifications(jobName, jobConfig, type, data) {
    const notifications = jobConfig.notifications?.[`on_${type}`] || [];
    
    for (const notification of notifications) {
      try {
        if (notification.type === 'email') {
          await this.sendEmailNotification(jobName, notification, type, data);
        } else if (notification.type === 'webhook') {
          await this.sendWebhookNotification(jobName, notification, type, data);
        } else if (notification.type === 'slack') {
          await this.sendSlackNotification(jobName, notification, type, data);
        }
      } catch (error) {
        this.logger.error(`Failed to send ${notification.type} notification for ${jobName}`, {
          error: error.message
        });
      }
    }
  }

  /**
   * Send email notification
   */
  async sendEmailNotification(jobName, notification, type, data) {
    if (!this.emailTransporter) {
      this.logger.warn('Email transporter not configured, skipping email notification');
      return;
    }
    
    const subject = notification.subject || `Job ${type}: ${jobName}`;
    const body = this.formatNotificationBody(jobName, type, data);
    
    await this.emailTransporter.sendMail({
      from: this.config.notifications.email.from_address,
      to: notification.recipients.join(', '),
      subject,
      text: body,
      html: body.replace(/\n/g, '<br>')
    });
    
    this.logger.info(`Email notification sent for ${jobName}`, {
      type,
      recipients: notification.recipients
    });
  }

  /**
   * Send webhook notification
   */
  async sendWebhookNotification(jobName, notification, type, data) {
    const payload = {
      job_name: jobName,
      status: type,
      timestamp: new Date().toISOString(),
      ...data
    };
    
    const url = this.interpolateEnvVars(notification.url);
    const headers = {
      'Content-Type': 'application/json',
      ...this.config.notifications?.webhook?.headers
    };
    
    await axios.post(url, payload, {
      headers: this.interpolateEnvVars(headers),
      timeout: this.parseTimeout(this.config.notifications?.webhook?.default_timeout || '10s')
    });
    
    this.logger.info(`Webhook notification sent for ${jobName}`, {
      type,
      url: url.replace(/\/\/.*@/, '//***@') // Hide credentials in logs
    });
  }

  /**
   * Send Slack notification
   */
  async sendSlackNotification(jobName, notification, type, data) {
    const slackConfig = this.config.notifications?.slack;
    if (!slackConfig?.webhook_url) {
      this.logger.warn('Slack webhook URL not configured');
      return;
    }
    
    const color = type === 'success' ? 'good' : 'danger';
    const emoji = type === 'success' ? ':white_check_mark:' : ':x:';
    
    const payload = {
      channel: slackConfig.channel,
      username: slackConfig.username || 'RSE Scheduler',
      attachments: [{
        color,
        title: `${emoji} Job ${type}: ${jobName}`,
        text: this.formatNotificationBody(jobName, type, data),
        timestamp: Math.floor(Date.now() / 1000)
      }]
    };
    
    await axios.post(this.interpolateEnvVars(slackConfig.webhook_url), payload);
    
    this.logger.info(`Slack notification sent for ${jobName}`, { type });
  }

  /**
   * Format notification body
   */
  formatNotificationBody(jobName, type, data) {
    let body = `Job: ${jobName}\nStatus: ${type.toUpperCase()}\nTime: ${new Date().toISOString()}\n`;
    
    if (data.duration) {
      body += `Duration: ${data.duration}s\n`;
    }
    
    if (data.error) {
      body += `Error: ${data.error}\n`;
      if (data.retryCount) {
        body += `Retry Count: ${data.retryCount}\n`;
      }
    }
    
    if (data.output && type === 'success') {
      body += `Output: ${data.output.substring(0, 500)}${data.output.length > 500 ? '...' : ''}\n`;
    }
    
    return body;
  }

  /**
   * Start health check monitoring
   */
  startHealthCheck() {
    setInterval(() => {
      this.performHealthCheck();
    }, 30000); // Every 30 seconds
  }

  /**
   * Perform health check
   */
  performHealthCheck() {
    try {
      const healthStatus = {
        timestamp: new Date().toISOString(),
        scheduler_running: true,
        active_jobs: this.runningJobs.size,
        registered_jobs: this.jobs.size,
        memory_usage: process.memoryUsage(),
        uptime: process.uptime()
      };
      
      // Check for stuck jobs (running for more than their timeout)
      const stuckJobs = [];
      for (const [jobName, jobState] of this.runningJobs) {
        const jobConfig = this.jobs.get(jobName)?.config;
        if (jobConfig) {
          const timeout = this.parseTimeout(jobConfig.timeout || '10m');
          const runningTime = Date.now() - jobState.startTime;
          if (runningTime > timeout * 1.5) { // 1.5x timeout threshold
            stuckJobs.push({ jobName, runningTime: runningTime / 1000 });
          }
        }
      }
      
      if (stuckJobs.length > 0) {
        this.logger.warn('Detected stuck jobs', { stuckJobs });
        healthStatus.stuck_jobs = stuckJobs;
        this.metrics.schedulerHealth.set(0);
      } else {
        this.metrics.schedulerHealth.set(1);
      }
      
      this.emit('health-check', healthStatus);
      
    } catch (error) {
      this.logger.error('Health check failed', { error: error.message });
      this.metrics.schedulerHealth.set(0);
    }
  }

  /**
   * Start metrics server for Prometheus
   */
  startMetricsServer() {
    const express = require('express');
    const app = express();
    const port = this.config.monitoring?.metrics?.port || 9090;
    
    app.get('/metrics', async (req, res) => {
      try {
        res.set('Content-Type', this.metrics.register.contentType);
        res.end(await this.metrics.register.metrics());
      } catch (error) {
        res.status(500).end(error.message);
      }
    });
    
    app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        active_jobs: this.runningJobs.size,
        registered_jobs: this.jobs.size
      });
    });
    
    app.listen(port, () => {
      this.logger.info(`Metrics server started on port ${port}`);
    });
  }

  /**
   * Parse timeout string to milliseconds
   */
  parseTimeout(timeoutStr) {
    const match = timeoutStr.match(/^(\d+)([smh])$/);
    if (!match) {
      throw new Error(`Invalid timeout format: ${timeoutStr}`);
    }
    
    const value = parseInt(match[1]);
    const unit = match[2];
    
    switch (unit) {
      case 's': return value * 1000;
      case 'm': return value * 60 * 1000;
      case 'h': return value * 60 * 60 * 1000;
      default: throw new Error(`Unknown timeout unit: ${unit}`);
    }
  }

  /**
   * Interpolate environment variables in strings
   */
  interpolateEnvVars(obj) {
    if (typeof obj === 'string') {
      return obj.replace(/\$\{([^}]+)\}/g, (match, varName) => {
        return process.env[varName] || match;
      });
    } else if (typeof obj === 'object' && obj !== null) {
      const result = {};
      for (const [key, value] of Object.entries(obj)) {
        result[key] = this.interpolateEnvVars(value);
      }
      return result;
    }
    return obj;
  }

  /**
   * Manually trigger a job
   */
  async triggerJob(jobName) {
    const jobInfo = this.jobs.get(jobName);
    if (!jobInfo) {
      throw new Error(`Job not found: ${jobName}`);
    }
    
    this.logger.info(`Manually triggering job: ${jobName}`);
    await this.executeJob(jobName, jobInfo.config);
  }

  /**
   * Get job status and history
   */
  getJobStatus(jobName) {
    const jobInfo = this.jobs.get(jobName);
    if (!jobInfo) {
      return null;
    }
    
    const runningJob = this.runningJobs.get(jobName);
    const history = this.jobHistory.get(jobName) || [];
    
    return {
      name: jobName,
      config: jobInfo.config,
      lastRun: jobInfo.lastRun,
      runCount: jobInfo.runCount,
      failureCount: jobInfo.failureCount,
      currentState: runningJob ? runningJob.state : 'idle',
      history: history.slice(-10) // Last 10 executions
    };
  }

  /**
   * Stop the scheduler
   */
  stop() {
    this.logger.info('Stopping RSE Scheduler Orchestrator');
    
    // Stop all cron jobs
    for (const [jobName, jobInfo] of this.jobs) {
      jobInfo.task.stop();
      this.logger.info(`Stopped job: ${jobName}`);
    }
    
    // Cancel running jobs
    for (const [jobName, jobState] of this.runningJobs) {
      jobState.state = this.JOB_STATES.CANCELLED;
      this.logger.info(`Cancelled running job: ${jobName}`);
    }
    
    this.jobs.clear();
    this.runningJobs.clear();
    
    this.emit('stopped');
    this.logger.info('RSE Scheduler Orchestrator stopped');
  }
}

// Export the class and create a default instance
module.exports = RSESchedulerOrchestrator;

// If this file is run directly, start the scheduler
if (require.main === module) {
  const scheduler = new RSESchedulerOrchestrator();
  
  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nReceived SIGINT, shutting down gracefully...');
    scheduler.stop();
    process.exit(0);
  });
  
  process.on('SIGTERM', () => {
    console.log('\nReceived SIGTERM, shutting down gracefully...');
    scheduler.stop();
    process.exit(0);
  });
  
  // Start the scheduler
  scheduler.start();
  
  // Log some example usage
  scheduler.on('started', () => {
    console.log('\n🚀 RSE Scheduler Orchestrator is running!');
    console.log('\n📊 Monitoring endpoints:');
    console.log('  - Health: http://localhost:9090/health');
    console.log('  - Metrics: http://localhost:9090/metrics');
    console.log('\n📝 Example API calls:');
    console.log('  - Trigger job: scheduler.triggerJob("theme_extraction")');
    console.log('  - Get status: scheduler.getJobStatus("news_ingestion")');
    console.log('\n🔧 Configuration loaded from: ./jobs.yaml');
  });
}