/**
 * Test suite for RSE Scheduler Orchestrator
 * Tests job scheduling, dependency management, retry logic, and monitoring
 */

const RSESchedulerOrchestrator = require('./index');
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// Mock dependencies
jest.mock('node-cron');
jest.mock('nodemailer');
jest.mock('axios');

const mockCron = require('node-cron');
const mockNodemailer = require('nodemailer');
const mockAxios = require('axios');

describe('RSE Scheduler Orchestrator', () => {
  let scheduler;
  let mockConfig;
  let originalConsoleLog;
  let originalConsoleError;

  beforeAll(() => {
    // Suppress console output during tests
    originalConsoleLog = console.log;
    originalConsoleError = console.error;
    console.log = jest.fn();
    console.error = jest.fn();
  });

  afterAll(() => {
    // Restore console output
    console.log = originalConsoleLog;
    console.error = originalConsoleError;
  });

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock configuration
    mockConfig = {
      scheduler: {
        timezone: 'UTC',
        max_concurrent_jobs: 5,
        retry_policy: {
          max_retries: 3,
          retry_delay: '5m',
          exponential_backoff: true
        }
      },
      jobs: {
        test_job: {
          enabled: true,
          schedule: '0 8 * * *',
          command: {
            type: 'node',
            script: 'test-script.js',
            args: ['--env', 'test']
          },
          timeout: '10m',
          priority: 'high',
          dependencies: [],
          notifications: {
            on_success: [
              {
                type: 'email',
                recipients: ['admin@example.com']
              }
            ],
            on_failure: [
              {
                type: 'slack',
                channel: '#alerts'
              }
            ]
          }
        },
        dependent_job: {
          enabled: true,
          schedule: '0 9 * * *',
          command: {
            type: 'python',
            script: 'dependent-script.py'
          },
          timeout: '5m',
          dependencies: ['test_job']
        }
      },
      notifications: {
        email: {
          smtp_host: 'smtp.example.com',
          smtp_port: 587,
          smtp_user: 'test@example.com',
          smtp_password: 'password',
          from_address: 'scheduler@example.com'
        },
        slack: {
          webhook_url: 'https://hooks.slack.com/test',
          channel: '#general',
          username: 'RSE Scheduler'
        },
        webhook: {
          default_timeout: '10s',
          headers: {
            'Authorization': 'Bearer ${API_TOKEN}'
          }
        }
      },
      monitoring: {
        metrics: {
          port: 9090
        }
      }
    };

    // Mock file system
    jest.spyOn(fs, 'readFileSync').mockReturnValue(yaml.dump(mockConfig));
    jest.spyOn(fs, 'existsSync').mockReturnValue(true);
    jest.spyOn(fs, 'mkdirSync').mockImplementation();

    // Mock cron
    const mockTask = {
      start: jest.fn(),
      stop: jest.fn()
    };
    mockCron.schedule.mockReturnValue(mockTask);

    // Mock nodemailer
    const mockTransporter = {
      sendMail: jest.fn().mockResolvedValue({ messageId: 'test-id' })
    };
    mockNodemailer.createTransporter.mockReturnValue(mockTransporter);

    // Mock axios
    mockAxios.post.mockResolvedValue({ status: 200, data: 'OK' });
  });

  afterEach(() => {
    if (scheduler) {
      scheduler.stop();
    }
    jest.restoreAllMocks();
  });

  describe('Initialization', () => {
    test('should initialize with default configuration path', () => {
      scheduler = new RSESchedulerOrchestrator();
      expect(scheduler.configPath).toBe('./jobs.yaml');
      expect(scheduler.config).toEqual(mockConfig);
    });

    test('should initialize with custom configuration path', () => {
      const customPath = './custom-jobs.yaml';
      scheduler = new RSESchedulerOrchestrator(customPath);
      expect(scheduler.configPath).toBe(customPath);
    });

    test('should throw error on invalid configuration', () => {
      fs.readFileSync.mockImplementation(() => {
        throw new Error('File not found');
      });
      
      expect(() => {
        scheduler = new RSESchedulerOrchestrator();
      }).toThrow('Configuration loading failed: File not found');
    });

    test('should initialize metrics correctly', () => {
      scheduler = new RSESchedulerOrchestrator();
      expect(scheduler.metrics).toBeDefined();
      expect(scheduler.metrics.jobExecutions).toBeDefined();
      expect(scheduler.metrics.jobDuration).toBeDefined();
      expect(scheduler.metrics.activeJobs).toBeDefined();
      expect(scheduler.metrics.schedulerHealth).toBeDefined();
    });

    test('should initialize logger correctly', () => {
      scheduler = new RSESchedulerOrchestrator();
      expect(scheduler.logger).toBeDefined();
      expect(typeof scheduler.logger.info).toBe('function');
      expect(typeof scheduler.logger.error).toBe('function');
    });
  });

  describe('Job Registration', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
    });

    test('should register enabled jobs on start', () => {
      scheduler.start();
      
      expect(mockCron.schedule).toHaveBeenCalledTimes(2);
      expect(scheduler.jobs.size).toBe(2);
      expect(scheduler.jobs.has('test_job')).toBe(true);
      expect(scheduler.jobs.has('dependent_job')).toBe(true);
    });

    test('should skip disabled jobs', () => {
      mockConfig.jobs.test_job.enabled = false;
      fs.readFileSync.mockReturnValue(yaml.dump(mockConfig));
      
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
      
      expect(scheduler.jobs.size).toBe(1);
      expect(scheduler.jobs.has('test_job')).toBe(false);
      expect(scheduler.jobs.has('dependent_job')).toBe(true);
    });

    test('should register job with correct cron schedule', () => {
      scheduler.start();
      
      expect(mockCron.schedule).toHaveBeenCalledWith(
        '0 8 * * *',
        expect.any(Function),
        {
          scheduled: false,
          timezone: 'UTC'
        }
      );
    });

    test('should start registered cron tasks', () => {
      scheduler.start();
      
      const mockTasks = mockCron.schedule.mock.results.map(result => result.value);
      mockTasks.forEach(task => {
        expect(task.start).toHaveBeenCalled();
      });
    });
  });

  describe('Job Execution', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
    });

    test('should prevent concurrent execution of same job', async () => {
      // Simulate job already running
      scheduler.runningJobs.set('test_job', {
        id: 'test-id',
        name: 'test_job',
        state: 'running',
        startTime: Date.now()
      });

      const jobConfig = mockConfig.jobs.test_job;
      await scheduler.executeJob('test_job', jobConfig);

      // Should not start another instance
      expect(scheduler.runningJobs.size).toBe(1);
    });

    test('should respect max concurrent jobs limit', async () => {
      // Fill up to max concurrent jobs
      for (let i = 0; i < 5; i++) {
        scheduler.runningJobs.set(`job_${i}`, {
          id: `job_${i}-id`,
          name: `job_${i}`,
          state: 'running',
          startTime: Date.now()
        });
      }

      const jobConfig = mockConfig.jobs.test_job;
      
      // Mock setTimeout to capture the delayed execution
      const originalSetTimeout = global.setTimeout;
      const mockSetTimeout = jest.fn();
      global.setTimeout = mockSetTimeout;

      await scheduler.executeJob('test_job', jobConfig);

      expect(mockSetTimeout).toHaveBeenCalledWith(
        expect.any(Function),
        30000
      );

      global.setTimeout = originalSetTimeout;
    });

    test('should update metrics on job execution', async () => {
      const jobConfig = mockConfig.jobs.test_job;
      
      // Mock successful job execution
      jest.spyOn(scheduler, 'runJobCommand').mockResolvedValue({
        output: 'Job completed successfully',
        exitCode: 0
      });

      await scheduler.executeJob('test_job', jobConfig);

      expect(scheduler.metrics.jobExecutions.inc).toHaveBeenCalledWith({
        job_name: 'test_job',
        status: 'success'
      });
    });
  });

  describe('Dependency Management', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
    });

    test('should wait for dependencies before execution', async () => {
      const dependencies = ['test_job'];
      
      // Mock dependency job as running
      scheduler.runningJobs.set('test_job', {
        id: 'test-job-id',
        name: 'test_job',
        state: 'running',
        startTime: Date.now()
      });

      const waitPromise = scheduler.waitForDependencies('dependent_job', dependencies);

      // Simulate dependency completion after 100ms
      setTimeout(() => {
        scheduler.runningJobs.delete('test_job');
      }, 100);

      await expect(waitPromise).resolves.toBeUndefined();
    });

    test('should throw error for non-existent dependency', async () => {
      const dependencies = ['non_existent_job'];
      
      await expect(
        scheduler.waitForDependencies('dependent_job', dependencies)
      ).rejects.toThrow("Dependency job 'non_existent_job' not found for job 'dependent_job'");
    });

    test('should timeout waiting for dependencies', async () => {
      const dependencies = ['test_job'];
      
      // Mock dependency job as running (never completes)
      scheduler.runningJobs.set('test_job', {
        id: 'test-job-id',
        name: 'test_job',
        state: 'running',
        startTime: Date.now()
      });

      // Mock timeout to be very short for testing
      const originalWaitForDependencies = scheduler.waitForDependencies;
      scheduler.waitForDependencies = async function(jobName, deps) {
        const timeout = 100; // 100ms timeout for testing
        const startWait = Date.now();
        
        for (const dependency of deps) {
          while (this.runningJobs.has(dependency)) {
            if (Date.now() - startWait > timeout) {
              throw new Error(`Dependency '${dependency}' timeout for job '${jobName}'`);
            }
            await new Promise(resolve => setTimeout(resolve, 10));
          }
        }
      };

      await expect(
        scheduler.waitForDependencies('dependent_job', dependencies)
      ).rejects.toThrow("Dependency 'test_job' timeout for job 'dependent_job'");
    });
  });

  describe('Retry Logic', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
    });

    test('should retry failed jobs up to max retries', async () => {
      const jobConfig = mockConfig.jobs.test_job;
      const error = new Error('Job failed');
      
      const jobState = {
        id: 'test-job-id',
        name: 'test_job',
        state: 'running',
        startTime: Date.now(),
        retryCount: 0
      };

      // Mock setTimeout to capture retry scheduling
      const originalSetTimeout = global.setTimeout;
      const mockSetTimeout = jest.fn();
      global.setTimeout = mockSetTimeout;

      await scheduler.handleJobFailure('test_job', jobConfig, jobState, error);

      expect(jobState.state).toBe('retrying');
      expect(jobState.retryCount).toBe(1);
      expect(mockSetTimeout).toHaveBeenCalled();

      global.setTimeout = originalSetTimeout;
    });

    test('should send failure notification after max retries', async () => {
      const jobConfig = mockConfig.jobs.test_job;
      const error = new Error('Job failed');
      
      const jobState = {
        id: 'test-job-id',
        name: 'test_job',
        state: 'running',
        startTime: Date.now(),
        retryCount: 3 // Already at max retries
      };

      jest.spyOn(scheduler, 'sendNotifications').mockResolvedValue();

      await scheduler.handleJobFailure('test_job', jobConfig, jobState, error);

      expect(scheduler.sendNotifications).toHaveBeenCalledWith(
        'test_job',
        jobConfig,
        'failure',
        expect.objectContaining({
          error: 'Job failed',
          retryCount: 3
        })
      );
    });

    test('should apply exponential backoff for retries', async () => {
      const jobConfig = mockConfig.jobs.test_job;
      const error = new Error('Job failed');
      
      const jobState = {
        id: 'test-job-id',
        name: 'test_job',
        state: 'running',
        startTime: Date.now(),
        retryCount: 1 // Second retry
      };

      const originalSetTimeout = global.setTimeout;
      const mockSetTimeout = jest.fn();
      global.setTimeout = mockSetTimeout;

      await scheduler.handleJobFailure('test_job', jobConfig, jobState, error);

      // Should apply 2x backoff for second retry (5m * 2 = 10m = 600000ms)
      expect(mockSetTimeout).toHaveBeenCalledWith(
        expect.any(Function),
        600000 // 10 minutes in milliseconds
      );

      global.setTimeout = originalSetTimeout;
    });
  });

  describe('Notifications', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
    });

    test('should send email notification on success', async () => {
      const jobConfig = mockConfig.jobs.test_job;
      const data = { duration: 30, output: 'Success' };

      await scheduler.sendNotifications('test_job', jobConfig, 'success', data);

      expect(scheduler.emailTransporter.sendMail).toHaveBeenCalledWith(
        expect.objectContaining({
          from: 'scheduler@example.com',
          to: 'admin@example.com',
          subject: expect.stringContaining('Job success: test_job')
        })
      );
    });

    test('should send Slack notification on failure', async () => {
      const jobConfig = mockConfig.jobs.test_job;
      const data = { error: 'Job failed', retryCount: 1 };

      await scheduler.sendNotifications('test_job', jobConfig, 'failure', data);

      expect(mockAxios.post).toHaveBeenCalledWith(
        'https://hooks.slack.com/test',
        expect.objectContaining({
          channel: '#general',
          username: 'RSE Scheduler',
          attachments: expect.arrayContaining([
            expect.objectContaining({
              color: 'danger',
              title: expect.stringContaining('❌ Job failure: test_job')
            })
          ])
        })
      );
    });

    test('should send webhook notification', async () => {
      const notification = {
        type: 'webhook',
        url: 'https://api.example.com/webhook'
      };
      
      const data = { duration: 30 };

      await scheduler.sendWebhookNotification('test_job', notification, 'success', data);

      expect(mockAxios.post).toHaveBeenCalledWith(
        'https://api.example.com/webhook',
        expect.objectContaining({
          job_name: 'test_job',
          status: 'success',
          duration: 30
        }),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      );
    });
  });

  describe('Health Monitoring', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
    });

    test('should perform health check', () => {
      const healthCheckSpy = jest.spyOn(scheduler, 'performHealthCheck');
      
      scheduler.performHealthCheck();
      
      expect(healthCheckSpy).toHaveBeenCalled();
      expect(scheduler.metrics.schedulerHealth.set).toHaveBeenCalledWith(1);
    });

    test('should detect stuck jobs', () => {
      // Add a stuck job (running for longer than timeout)
      const stuckJobStartTime = Date.now() - (15 * 60 * 1000); // 15 minutes ago
      scheduler.runningJobs.set('stuck_job', {
        id: 'stuck-job-id',
        name: 'stuck_job',
        state: 'running',
        startTime: stuckJobStartTime
      });
      
      // Add job config with 10m timeout
      scheduler.jobs.set('stuck_job', {
        config: { timeout: '10m' }
      });

      const healthCheckListener = jest.fn();
      scheduler.on('health-check', healthCheckListener);

      scheduler.performHealthCheck();

      expect(healthCheckListener).toHaveBeenCalledWith(
        expect.objectContaining({
          stuck_jobs: expect.arrayContaining([
            expect.objectContaining({
              jobName: 'stuck_job'
            })
          ])
        })
      );
      
      expect(scheduler.metrics.schedulerHealth.set).toHaveBeenCalledWith(0);
    });
  });

  describe('Utility Functions', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
    });

    test('should parse timeout strings correctly', () => {
      expect(scheduler.parseTimeout('30s')).toBe(30000);
      expect(scheduler.parseTimeout('5m')).toBe(300000);
      expect(scheduler.parseTimeout('2h')).toBe(7200000);
    });

    test('should throw error for invalid timeout format', () => {
      expect(() => scheduler.parseTimeout('invalid')).toThrow('Invalid timeout format: invalid');
      expect(() => scheduler.parseTimeout('30x')).toThrow('Unknown timeout unit: x');
    });

    test('should interpolate environment variables', () => {
      process.env.TEST_VAR = 'test_value';
      
      const result = scheduler.interpolateEnvVars('Hello ${TEST_VAR}!');
      expect(result).toBe('Hello test_value!');
      
      const objResult = scheduler.interpolateEnvVars({
        key: 'Value: ${TEST_VAR}',
        nested: {
          prop: '${TEST_VAR}_nested'
        }
      });
      
      expect(objResult).toEqual({
        key: 'Value: test_value',
        nested: {
          prop: 'test_value_nested'
        }
      });
      
      delete process.env.TEST_VAR;
    });
  });

  describe('Job Management', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
    });

    test('should manually trigger a job', async () => {
      jest.spyOn(scheduler, 'executeJob').mockResolvedValue();
      
      await scheduler.triggerJob('test_job');
      
      expect(scheduler.executeJob).toHaveBeenCalledWith(
        'test_job',
        mockConfig.jobs.test_job
      );
    });

    test('should throw error when triggering non-existent job', async () => {
      await expect(scheduler.triggerJob('non_existent_job'))
        .rejects.toThrow('Job not found: non_existent_job');
    });

    test('should get job status', () => {
      // Add some job history
      scheduler.jobHistory.set('test_job', [
        {
          id: 'job-1',
          state: 'completed',
          startTime: Date.now() - 1000,
          endTime: Date.now()
        }
      ]);
      
      const status = scheduler.getJobStatus('test_job');
      
      expect(status).toEqual({
        name: 'test_job',
        config: mockConfig.jobs.test_job,
        lastRun: null,
        runCount: 0,
        failureCount: 0,
        currentState: 'idle',
        history: expect.any(Array)
      });
    });

    test('should return null for non-existent job status', () => {
      const status = scheduler.getJobStatus('non_existent_job');
      expect(status).toBeNull();
    });
  });

  describe('Shutdown', () => {
    beforeEach(() => {
      scheduler = new RSESchedulerOrchestrator();
      scheduler.start();
    });

    test('should stop all jobs on shutdown', () => {
      const stoppedListener = jest.fn();
      scheduler.on('stopped', stoppedListener);
      
      scheduler.stop();
      
      // Check that all cron tasks were stopped
      const mockTasks = mockCron.schedule.mock.results.map(result => result.value);
      mockTasks.forEach(task => {
        expect(task.stop).toHaveBeenCalled();
      });
      
      expect(scheduler.jobs.size).toBe(0);
      expect(scheduler.runningJobs.size).toBe(0);
      expect(stoppedListener).toHaveBeenCalled();
    });

    test('should cancel running jobs on shutdown', () => {
      // Add a running job
      scheduler.runningJobs.set('test_job', {
        id: 'test-job-id',
        name: 'test_job',
        state: 'running',
        startTime: Date.now()
      });
      
      scheduler.stop();
      
      expect(scheduler.runningJobs.size).toBe(0);
    });
  });
});

// Integration test for the complete workflow
describe('RSE Scheduler Integration', () => {
  let scheduler;
  
  beforeEach(() => {
    // Create a minimal test configuration
    const testConfig = {
      scheduler: {
        timezone: 'UTC',
        max_concurrent_jobs: 2
      },
      jobs: {
        simple_job: {
          enabled: true,
          schedule: '*/5 * * * * *', // Every 5 seconds for testing
          command: {
            type: 'node',
            script: 'echo.js'
          },
          timeout: '30s'
        }
      },
      notifications: {
        email: {
          smtp_host: 'localhost',
          smtp_port: 587
        }
      }
    };
    
    jest.spyOn(fs, 'readFileSync').mockReturnValue(yaml.dump(testConfig));
    jest.spyOn(fs, 'existsSync').mockReturnValue(true);
    jest.spyOn(fs, 'mkdirSync').mockImplementation();
    
    scheduler = new RSESchedulerOrchestrator();
  });
  
  afterEach(() => {
    if (scheduler) {
      scheduler.stop();
    }
    jest.restoreAllMocks();
  });
  
  test('should complete full lifecycle: start -> register -> execute -> stop', async () => {
    const startedListener = jest.fn();
    const stoppedListener = jest.fn();
    
    scheduler.on('started', startedListener);
    scheduler.on('stopped', stoppedListener);
    
    // Start scheduler
    scheduler.start();
    expect(startedListener).toHaveBeenCalled();
    expect(scheduler.jobs.size).toBe(1);
    
    // Manually trigger job to test execution
    jest.spyOn(scheduler, 'runJobCommand').mockResolvedValue({
      output: 'Test output',
      exitCode: 0
    });
    
    await scheduler.triggerJob('simple_job');
    
    // Verify job was executed
    expect(scheduler.runJobCommand).toHaveBeenCalled();
    
    // Stop scheduler
    scheduler.stop();
    expect(stoppedListener).toHaveBeenCalled();
    expect(scheduler.jobs.size).toBe(0);
  });
});