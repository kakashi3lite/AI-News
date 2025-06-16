import { NextResponse } from 'next/server';

// GET /api/monitoring/alerts/[alertId]
export async function GET(request, { params }) {
  try {
    const { alertId } = params;
    
    // Mock alert data - in real implementation, fetch from database
    const alerts = {
      'alert_001': {
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
        escalationPolicy: 'on_call_engineer',
        history: [
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 8).toISOString(),
            action: 'created',
            user: 'system',
            details: 'Alert triggered due to memory usage exceeding threshold'
          }
        ],
        relatedAlerts: ['alert_005'],
        metrics: {
          duration: 8,
          peakValue: 87.5,
          averageValue: 84.1,
          occurrences: 1
        }
      },
      'alert_002': {
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
        escalationPolicy: 'senior_engineer',
        history: [
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
            action: 'created',
            user: 'system',
            details: 'Alert triggered due to API response time spike'
          },
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
            action: 'acknowledged',
            user: 'admin@example.com',
            details: 'Investigating the issue'
          }
        ],
        relatedAlerts: [],
        metrics: {
          duration: 3,
          peakValue: 1200,
          averageValue: 945,
          occurrences: 1
        }
      },
      'alert_003': {
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
        incidents: ['incident_001'],
        history: [
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 1).toISOString(),
            action: 'created',
            user: 'system',
            details: 'Critical alert: Database connection pool exhausted'
          }
        ],
        relatedAlerts: ['alert_002'],
        metrics: {
          duration: 1,
          peakValue: 100,
          averageValue: 98,
          occurrences: 1
        }
      }
    };
    
    const alert = alerts[alertId];
    
    if (!alert) {
      return NextResponse.json(
        { error: 'Alert not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json(alert);
  } catch (error) {
    console.error(`[/api/monitoring/alerts/${params.alertId}] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// PATCH /api/monitoring/alerts/[alertId]
export async function PATCH(request, { params }) {
  try {
    const { alertId } = params;
    const body = await request.json();
    const { action, user, comment, ...updateData } = body;
    
    // Handle different actions
    switch (action) {
      case 'acknowledge':
        const acknowledgeResult = {
          id: alertId,
          status: 'acknowledged',
          acknowledged: true,
          acknowledgedBy: user || 'unknown',
          acknowledgedAt: new Date().toISOString(),
          comment: comment || '',
          message: 'Alert acknowledged successfully'
        };
        
        console.log(`Acknowledging alert ${alertId}:`, acknowledgeResult);
        return NextResponse.json(acknowledgeResult);
        
      case 'resolve':
        const resolveResult = {
          id: alertId,
          status: 'resolved',
          resolvedBy: user || 'unknown',
          resolvedAt: new Date().toISOString(),
          comment: comment || '',
          message: 'Alert resolved successfully'
        };
        
        console.log(`Resolving alert ${alertId}:`, resolveResult);
        return NextResponse.json(resolveResult);
        
      case 'escalate':
        const escalateResult = {
          id: alertId,
          escalated: true,
          escalatedBy: user || 'unknown',
          escalatedAt: new Date().toISOString(),
          escalationLevel: updateData.escalationLevel || 'next_tier',
          comment: comment || '',
          message: 'Alert escalated successfully'
        };
        
        console.log(`Escalating alert ${alertId}:`, escalateResult);
        return NextResponse.json(escalateResult);
        
      case 'snooze':
        const snoozeDuration = updateData.snoozeDuration || 30; // minutes
        const snoozeUntil = new Date(Date.now() + snoozeDuration * 60 * 1000);
        
        const snoozeResult = {
          id: alertId,
          status: 'snoozed',
          snoozedBy: user || 'unknown',
          snoozedAt: new Date().toISOString(),
          snoozeUntil: snoozeUntil.toISOString(),
          snoozeDuration,
          comment: comment || '',
          message: `Alert snoozed for ${snoozeDuration} minutes`
        };
        
        console.log(`Snoozing alert ${alertId}:`, snoozeResult);
        return NextResponse.json(snoozeResult);
        
      case 'update':
        const updateResult = {
          id: alertId,
          ...updateData,
          updatedBy: user || 'unknown',
          updatedAt: new Date().toISOString(),
          message: 'Alert updated successfully'
        };
        
        console.log(`Updating alert ${alertId}:`, updateResult);
        return NextResponse.json(updateResult);
        
      case 'add_comment':
        const commentResult = {
          id: alertId,
          comment: {
            id: `comment_${Date.now()}`,
            text: comment || '',
            author: user || 'unknown',
            timestamp: new Date().toISOString()
          },
          message: 'Comment added successfully'
        };
        
        console.log(`Adding comment to alert ${alertId}:`, commentResult);
        return NextResponse.json(commentResult);
        
      default:
        return NextResponse.json(
          { error: 'Invalid action. Supported actions: acknowledge, resolve, escalate, snooze, update, add_comment' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error(`[/api/monitoring/alerts/${params.alertId}] PATCH Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// DELETE /api/monitoring/alerts/[alertId]
export async function DELETE(request, { params }) {
  try {
    const { alertId } = params;
    const { searchParams } = new URL(request.url);
    const force = searchParams.get('force') === 'true';
    
    // In real implementation, check if alert can be deleted
    // (e.g., only resolved alerts, user has permissions, etc.)
    
    if (!force) {
      // Check if alert is resolved before allowing deletion
      // This is a mock check - in real implementation, fetch from database
      const mockAlertStatus = 'active'; // This would come from database
      
      if (mockAlertStatus !== 'resolved') {
        return NextResponse.json(
          { error: 'Only resolved alerts can be deleted. Use force=true to override.' },
          { status: 400 }
        );
      }
    }
    
    console.log(`Deleting alert ${alertId} (force: ${force})`);
    
    return NextResponse.json({
      success: true,
      message: 'Alert deleted successfully',
      deletedId: alertId,
      forced: force
    });
  } catch (error) {
    console.error(`[/api/monitoring/alerts/${params.alertId}] DELETE Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}