/**
 * AI News Dashboard - Real-time Monitoring Dashboard
 * Context-aware system monitoring with Dr. Vega's solo-founder approach
 * Built by Dr. Phoenix "SoloSprint" Vega
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useContextAware } from '../contexts/ContextProvider';
import { ExperimentationEngine } from './ExperimentationEngine';

// Monitoring Dashboard Component
const MonitoringDashboard = ({ isOpen, onClose }) => {
  const { contextState, trackInteraction } = useContextAware();
  const [healthData, setHealthData] = useState(null);
  const [metrics, setMetrics] = useState({
    realTime: {},
    historical: [],
    alerts: []
  });
  const [activeTab, setActiveTab] = useState('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const intervalRef = useRef(null);
  const wsRef = useRef(null);

  // Health check and metrics fetching
  const fetchHealthData = async () => {
    try {
      const response = await fetch('/api/health');
      const data = await response.json();
      setHealthData(data);
      
      // Update metrics
      setMetrics(prev => ({
        ...prev,
        realTime: {
          status: data.status,
          responseTime: data.performance?.responseTime || 0,
          uptime: data.uptime,
          memoryUsage: data.performance?.cpuUsage || {},
          activeFeatures: Object.values(data.features || {}).filter(f => f.enabled).length,
          timestamp: Date.now()
        },
        historical: [...prev.historical.slice(-50), {
          timestamp: Date.now(),
          status: data.status,
          responseTime: data.performance?.responseTime || 0,
          memoryUsage: data.checks?.memory?.details?.heapUsed || 0
        }]
      }));
      
      // Check for alerts
      checkForAlerts(data);
      
    } catch (error) {
      console.error('Failed to fetch health data:', error);
      setHealthData({
        status: 'unhealthy',
        error: 'Failed to connect to health endpoint',
        timestamp: new Date().toISOString()
      });
    }
  };

  // Alert system
  const checkForAlerts = (data) => {
    const newAlerts = [];
    
    // Check for critical issues
    if (data.status === 'unhealthy') {
      newAlerts.push({
        id: `critical-${Date.now()}`,
        type: 'critical',
        title: 'System Unhealthy',
        message: data.message || 'Critical services are down',
        timestamp: Date.now()
      });
    }
    
    // Check memory usage
    const memoryUsage = data.checks?.memory?.details;
    if (memoryUsage && (memoryUsage.heapUsed / memoryUsage.heapTotal) > 0.8) {
      newAlerts.push({
        id: `memory-${Date.now()}`,
        type: 'warning',
        title: 'High Memory Usage',
        message: `Memory usage at ${Math.round((memoryUsage.heapUsed / memoryUsage.heapTotal) * 100)}%`,
        timestamp: Date.now()
      });
    }
    
    // Check response time
    if (data.performance?.responseTime > 1000) {
      newAlerts.push({
        id: `performance-${Date.now()}`,
        type: 'warning',
        title: 'Slow Response Time',
        message: `Health check took ${data.performance.responseTime}ms`,
        timestamp: Date.now()
      });
    }
    
    if (newAlerts.length > 0) {
      setMetrics(prev => ({
        ...prev,
        alerts: [...prev.alerts.slice(-10), ...newAlerts]
      }));
    }
  };

  // WebSocket connection for real-time updates
  const setupWebSocket = () => {
    if (typeof window !== 'undefined' && 'WebSocket' in window) {
      try {
        wsRef.current = new WebSocket(`ws://localhost:3001/monitoring`);
        
        wsRef.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'metrics') {
            setMetrics(prev => ({
              ...prev,
              realTime: { ...prev.realTime, ...data.payload }
            }));
          }
        };
        
        wsRef.current.onerror = () => {
          console.log('WebSocket connection failed, falling back to polling');
        };
      } catch (error) {
        console.log('WebSocket not available, using polling');
      }
    }
  };

  // Effects
  useEffect(() => {
    if (isOpen) {
      fetchHealthData();
      setupWebSocket();
      
      if (autoRefresh) {
        intervalRef.current = setInterval(fetchHealthData, refreshInterval);
      }
      
      trackInteraction('monitoring_dashboard_opened', {
        timestamp: Date.now(),
        context: contextState.session
      });
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isOpen, autoRefresh, refreshInterval]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (!isOpen) return;
      
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'r' && e.ctrlKey) {
        e.preventDefault();
        fetchHealthData();
      } else if (e.key >= '1' && e.key <= '4') {
        const tabs = ['overview', 'services', 'performance', 'experiments'];
        setActiveTab(tabs[parseInt(e.key) - 1]);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isOpen, onClose]);

  // Status indicator component
  const StatusIndicator = ({ status, size = 'sm' }) => {
    const colors = {
      healthy: 'bg-green-500',
      degraded: 'bg-yellow-500',
      unhealthy: 'bg-red-500'
    };
    
    const sizes = {
      sm: 'w-2 h-2',
      md: 'w-3 h-3',
      lg: 'w-4 h-4'
    };
    
    return (
      <div className={`rounded-full ${colors[status] || 'bg-gray-500'} ${sizes[size]} animate-pulse`} />
    );
  };

  // Metric card component
  const MetricCard = ({ title, value, unit, trend, status = 'healthy' }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700"
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</h3>
        <StatusIndicator status={status} />
      </div>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {value}
            {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
          </div>
          {trend && (
            <div className={`text-xs ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {trend > 0 ? '↗' : '↘'} {Math.abs(trend)}%
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );

  // Chart component (simplified)
  const SimpleChart = ({ data, height = 60 }) => {
    if (!data || data.length === 0) return null;
    
    const max = Math.max(...data.map(d => d.value));
    const min = Math.min(...data.map(d => d.value));
    const range = max - min || 1;
    
    return (
      <div className="flex items-end space-x-1" style={{ height }}>
        {data.slice(-20).map((point, index) => {
          const height = ((point.value - min) / range) * 50 + 10;
          return (
            <div
              key={index}
              className="bg-blue-500 rounded-t"
              style={{
                height: `${height}px`,
                width: '4px',
                opacity: 0.7 + (index / data.length) * 0.3
              }}
            />
          );
        })}
      </div>
    );
  };

  // Alert component
  const AlertItem = ({ alert, onDismiss }) => {
    const alertColors = {
      critical: 'border-red-500 bg-red-50 dark:bg-red-900/20',
      warning: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20',
      info: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
    };
    
    return (
      <motion.div
        initial={{ opacity: 0, x: 300 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 300 }}
        className={`border-l-4 p-3 rounded ${alertColors[alert.type]} mb-2`}
      >
        <div className="flex justify-between items-start">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white">{alert.title}</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">{alert.message}</p>
            <p className="text-xs text-gray-500 mt-1">
              {new Date(alert.timestamp).toLocaleTimeString()}
            </p>
          </div>
          <button
            onClick={() => onDismiss(alert.id)}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            ×
          </button>
        </div>
      </motion.div>
    );
  };

  // Tab content renderers
  const renderOverview = () => (
    <div className="space-y-6">
      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="System Status"
          value={healthData?.status || 'Unknown'}
          status={healthData?.status}
        />
        <MetricCard
          title="Uptime"
          value={healthData?.uptime ? Math.floor(healthData.uptime / 3600) : 0}
          unit="hours"
          status="healthy"
        />
        <MetricCard
          title="Response Time"
          value={metrics.realTime.responseTime || 0}
          unit="ms"
          status={metrics.realTime.responseTime > 1000 ? 'warning' : 'healthy'}
        />
        <MetricCard
          title="Active Features"
          value={metrics.realTime.activeFeatures || 0}
          status="healthy"
        />
      </div>
      
      {/* Performance Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Response Time Trend</h3>
        <SimpleChart
          data={metrics.historical.map(h => ({ value: h.responseTime, timestamp: h.timestamp }))}
        />
      </div>
      
      {/* Recent Alerts */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Recent Alerts</h3>
        <AnimatePresence>
          {metrics.alerts.slice(-5).map(alert => (
            <AlertItem
              key={alert.id}
              alert={alert}
              onDismiss={(id) => {
                setMetrics(prev => ({
                  ...prev,
                  alerts: prev.alerts.filter(a => a.id !== id)
                }));
              }}
            />
          ))}
        </AnimatePresence>
        {metrics.alerts.length === 0 && (
          <p className="text-gray-500 text-center py-4">No recent alerts</p>
        )}
      </div>
    </div>
  );

  const renderServices = () => (
    <div className="space-y-4">
      {healthData?.checks && Object.entries(healthData.checks).map(([service, check]) => (
        <motion.div
          key={service}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium capitalize">{service.replace('_', ' ')}</h3>
            <StatusIndicator status={check.status} size="md" />
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            <p>Status: <span className="capitalize">{check.status}</span></p>
            {check.responseTime && (
              <p>Response Time: {check.responseTime}ms</p>
            )}
            {check.error && (
              <p className="text-red-600 dark:text-red-400">Error: {check.error}</p>
            )}
            {check.details && (
              <div className="mt-2">
                <p className="font-medium">Details:</p>
                <pre className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded mt-1 overflow-auto">
                  {JSON.stringify(check.details, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );

  const renderPerformance = () => (
    <div className="space-y-6">
      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="Memory Usage"
          value={healthData?.checks?.memory?.details?.heapUsed || 0}
          unit="MB"
          status={healthData?.checks?.memory?.status}
        />
        <MetricCard
          title="Event Loop Lag"
          value={healthData?.performance?.eventLoopLag || 0}
          unit="ms"
          status={healthData?.performance?.eventLoopLag > 100 ? 'warning' : 'healthy'}
        />
        <MetricCard
          title="CPU Usage"
          value={healthData?.performance?.cpuUsage?.user || 0}
          unit="μs"
          status="healthy"
        />
      </div>
      
      {/* System Information */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">System Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <p><strong>Node Version:</strong> {healthData?.performance?.nodeVersion}</p>
            <p><strong>Platform:</strong> {healthData?.performance?.platform}</p>
            <p><strong>Architecture:</strong> {healthData?.performance?.architecture}</p>
          </div>
          <div>
            <p><strong>Environment:</strong> {healthData?.environment}</p>
            <p><strong>Version:</strong> {healthData?.version}</p>
            <p><strong>Uptime:</strong> {healthData?.uptime ? Math.floor(healthData.uptime / 3600) : 0}h</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderExperiments = () => (
    <div className="h-full">
      <ExperimentationEngine />
    </div>
  );

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl w-full max-w-6xl h-full max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-4">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                System Monitoring
              </h2>
              <StatusIndicator status={healthData?.status} size="lg" />
              <span className="text-sm text-gray-500">
                Last updated: {healthData?.timestamp ? new Date(healthData.timestamp).toLocaleTimeString() : 'Never'}
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Auto-refresh toggle */}
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="rounded"
                />
                <span>Auto-refresh</span>
              </label>
              
              {/* Refresh button */}
              <button
                onClick={fetchHealthData}
                className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              >
                Refresh
              </button>
              
              {/* Close button */}
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
          
          {/* Tabs */}
          <div className="flex border-b border-gray-200 dark:border-gray-700">
            {[
              { id: 'overview', label: 'Overview', key: '1' },
              { id: 'services', label: 'Services', key: '2' },
              { id: 'performance', label: 'Performance', key: '3' },
              { id: 'experiments', label: 'Experiments', key: '4' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                {tab.label}
                <span className="ml-2 text-xs text-gray-400">({tab.key})</span>
              </button>
            ))}
          </div>
          
          {/* Content */}
          <div className="flex-1 overflow-auto p-6">
            {activeTab === 'overview' && renderOverview()}
            {activeTab === 'services' && renderServices()}
            {activeTab === 'performance' && renderPerformance()}
            {activeTab === 'experiments' && renderExperiments()}
          </div>
          
          {/* Footer */}
          <div className="border-t border-gray-200 dark:border-gray-700 px-6 py-3 bg-gray-50 dark:bg-gray-800">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <div>
                Keyboard shortcuts: ESC (close), Ctrl+R (refresh), 1-4 (switch tabs)
              </div>
              <div>
                Built by Dr. Phoenix "SoloSprint" Vega • Context-Aware Monitoring
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default MonitoringDashboard;