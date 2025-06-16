import { NextResponse } from 'next/server';

// GET /api/monitoring/alerts
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status'); // active, resolved, acknowledged
    const severity = searchParams.get('severity'); // low, medium, high, critical
    const type = searchParams.get('type'); // info, warning, error, critical
    const limit = parseInt(searchParams.get('limit')) || 50;
    const offset = parseInt(searchParams.get('offset')) || 0;
    
    // Mock alerts data
    const allAlerts = [
      {
        id: 'alert_001',
        type: 'warning',
        severity: 'medium',
        title: 'High Memory Usage',
        description: 'Memory usage has exceeded 80% for the last 10 minutes',
        metric: 'system.memory.usage',
        threshold: 80,
        currentValue: 85.2,
        unit: '%',
        timestamp: new Date(Date.now() - 1000 * 60 * 8).toISOString(),
        status: 'active',
        acknowledged: false,
        source: 'system_monitor',
        tags: ['memory', 'system', 'performance'],
        affectedServices: ['api', 'database'],
        runbook: 'https://docs.example.com/runbooks/high-memory-usage',
        escalationPolicy: 'on_call_engineer'
      },
      {
        id: 'alert_002',
        type: 'error',
        severity: 'high',
        title: 'API Response Time Spike',
        description: 'API response time has increased by 150% in the last 5 minutes',
        metric: 'performance.api.responseTime',
        threshold: 500,
        currentValue: 890,
        unit: 'ms',
        timestamp: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
        status: 'acknowledged',
        acknowledged: true,
        acknowledgedBy: 'admin@example.com',
        acknowledgedAt: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
        source: 'apm_monitor',
        tags: ['api', 'performance', 'latency'],
        affectedServices: ['api'],
        runbook: 'https://docs.example.com/runbooks/api-latency',
        escalationPolicy: 'senior_engineer'
      },
      {
        id: 'alert_003',
        type: 'critical',
        severity: 'critical',
        title: 'Database Connection Pool Exhausted',
        description: 'All database connections are in use, new requests are being queued',
        metric: 'database.connections.active',
        threshold: 95,
        currentValue: 100,
        unit: 'connections',
        timestamp: new Date(Date.now() - 1000 * 60 * 1).toISOString(),
        status: 'active',
        acknowledged: false,
        source: 'database_monitor',
        tags: ['database', 'connections', 'critical'],
        affectedServices: ['api', 'database'],
        runbook: 'https://docs.example.com/runbooks/db-connection-pool',
        escalationPolicy: 'immediate_page',
        incidents: ['incident_001']
      },
      {
        id: 'alert_004',
        type: 'info',
        severity: 'low',
        title: 'Disk Space Warning',
        description: 'Disk usage approaching 30% threshold',
        metric: 'system.disk.usage',
        threshold: 30,
        currentValue: 28.5,
        unit: '%',
        timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
        status: 'resolved',
        acknowledged: true,
        acknowledgedBy: 'system@example.com',
        acknowledgedAt: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
        resolvedAt: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
        source: 'system_monitor',
        tags: ['disk', 'storage', 'capacity'],
        affectedServices: [],
        runbook: 'https://docs.example.com/runbooks/disk-cleanup'
      },
      {
        id: 'alert_005',
        type: 'warning',
        severity: 'medium',
        title: 'High Error Rate',
        description: 'HTTP 5xx error rate has exceeded 2% in the last 10 minutes',
        metric: 'performance.errors.rate',
        threshold: 2,
        currentValue: 3.8,
        unit: '%',
        timestamp: new Date(Date.now() - 1000 * 60 * 6).toISOString(),
        status: 'active',
        acknowledged: true,
        acknowledgedBy: 'devops@example.com',
        acknowledgedAt: new Date(Date.now() - 1000 * 60 * 4).toISOString(),
        source: 'error_monitor',
        tags: ['errors', 'http', 'reliability'],
        affectedServices: ['api'],
        runbook: 'https://docs.example.com/runbooks/high-error-rate',
        escalationPolicy: 'on_call_engineer'
      },
      {
        id: 'alert_006',
        type: 'warning',
        severity: 'medium',
        title: 'Cache Hit Rate Low',
        description: 'Redis cache hit rate has dropped below 85%',
        metric: 'cache.hitRate',
        threshold: 85,
        currentValue: 78.2,
        unit: '%',
        timestamp: new Date(Date.now() - 1000 * 60 * 12).toISOString(),
        status: 'active',
        acknowledged: false,
        source: 'cache_monitor',
        tags: ['cache', 'performance', 'redis'],
        affectedServices: ['cache'],
        runbook: 'https://docs.example.com/runbooks/cache-optimization'
      },
      {
        id: 'alert_007',
        type: 'info',
        severity: 'low',
        title: 'Scheduled Maintenance Reminder',
        description: 'Database maintenance window scheduled for tonight at 2 AM UTC',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
        status: 'active',
        acknowledged: true,
        acknowledgedBy: 'maintenance@example.com',
        acknowledgedAt: new Date(Date.now() - 1000 * 60 * 60 * 1).toISOString(),
        source: 'maintenance_scheduler',
        tags: ['maintenance', 'scheduled', 'database'],
        affectedServices: ['database'],
        maintenanceWindow: {
          start: new Date(Date.now() + 1000 * 60 * 60 * 6).toISOString(),
          end: new Date(Date.now() + 1000 * 60 * 60 * 8).toISOString(),
          duration: '2 hours'
        }
      },
      {
        id: 'alert_008',
        type: 'error',
        severity: 'high',
        title: 'SSL Certificate Expiring Soon',
        description: 'SSL certificate for api.example.com expires in 7 days',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
        status: 'active',
        acknowledged: false,
        source: 'ssl_monitor',
        tags: ['ssl', 'certificate', 'security'],
        affectedServices: ['api', 'cdn'],
        runbook: 'https://docs.example.com/runbooks/ssl-renewal',
        escalationPolicy: 'security_team',
        expirationDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * 7).toISOString()
      }
    ];
    
    // Filter alerts based on query parameters
    let alerts = allAlerts;
    
    if (status) {
      alerts = alerts.filter(alert => alert.status === status);
    }
    
    if (severity) {
      alerts = alerts.filter(alert => alert.severity === severity);
    }
    
    if (type) {
      alerts = alerts.filter(alert => alert.type === type);
    }
    
    // Sort by timestamp (newest first)
    alerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    // Apply pagination
    const paginatedAlerts = alerts.slice(offset, offset + limit);
    
    // Calculate summary statistics
    const stats = {
      total: alerts.length,
      byStatus: {
        active: alerts.filter(alert => alert.status === 'active').length,
        acknowledged: alerts.filter(alert => alert.status === 'acknowledged').length,
        resolved: alerts.filter(alert => alert.status === 'resolved').length
      },
      bySeverity: {
        low: alerts.filter(alert => alert.severity === 'low').length,
        medium: alerts.filter(alert => alert.severity === 'medium').length,
        high: alerts.filter(alert => alert.severity === 'high').length,
        critical: alerts.filter(alert => alert.severity === 'critical').length
      },
      byType: {
        info: alerts.filter(alert => alert.type === 'info').length,
        warning: alerts.filter(alert => alert.type === 'warning').length,
        error: alerts.filter(alert => alert.type === 'error').length,
        critical: alerts.filter(alert => alert.type === 'critical').length
      }
    };
    
    const response = {
      alerts: paginatedAlerts,
      stats,
      pagination: {
        total: alerts.length,
        limit,
        offset,
        hasMore: offset + limit < alerts.length
      },
      timestamp: new Date().toISOString()
    };
    
    return NextResponse.json(response);
  } catch (error) {
    console.error('[/api/monitoring/alerts] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/monitoring/alerts
export async function POST(request) {
  try {
    const body = await request.json();
    const {
      title,
      description,
      type,
      severity,
      metric,
      threshold,
      currentValue,
      source,
      tags,
      affectedServices
    } = body;
    
    // Validate required fields
    if (!title || !description || !type || !severity) {
      return NextResponse.json(
        { error: 'Missing required fields: title, description, type, severity' },
        { status: 400 }
      );
    }
    
    // Generate new alert ID
    const alertId = `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Create new alert
    const newAlert = {
      id: alertId,
      title,
      description,
      type,
      severity,
      metric: metric || null,
      threshold: threshold || null,
      currentValue: currentValue || null,
      timestamp: new Date().toISOString(),
      status: 'active',
      acknowledged: false,
      source: source || 'api',
      tags: tags || [],
      affectedServices: affectedServices || []
    };
    
    console.log('Created new alert:', newAlert);
    
    // In real implementation, save to database and trigger notifications
    
    return NextResponse.json({
      success: true,
      alert: newAlert,
      message: 'Alert created successfully'
    }, { status: 201 });
  } catch (error) {
    console.error('[/api/monitoring/alerts] POST Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}