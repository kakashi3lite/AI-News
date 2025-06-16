import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BarChart3, TrendingUp, Users, Zap, Settings, Eye, Target, Clock } from 'lucide-react';
import { useContextAware } from '../contexts/ContextProvider';

// Experiment Configuration
const EXPERIMENTS = {
  'context-search-triggers': {
    name: 'Context-Aware Search Triggers',
    description: 'Test different triggers for showing search suggestions',
    variants: {
      'time-based': {
        name: 'Time-Based Triggers',
        config: { triggerType: 'time', threshold: 30000 }, // 30 seconds
        weight: 0.33
      },
      'behavior-based': {
        name: 'Behavior-Based Triggers',
        config: { triggerType: 'behavior', threshold: 3 }, // 3 interactions
        weight: 0.33
      },
      'hybrid': {
        name: 'Hybrid Triggers',
        config: { triggerType: 'hybrid', timeThreshold: 20000, behaviorThreshold: 2 },
        weight: 0.34
      }
    },
    metrics: ['engagement_rate', 'search_completion', 'user_satisfaction'],
    status: 'active'
  },
  'ai-skill-presentation': {
    name: 'AI Skill Presentation Style',
    description: 'Test different ways to present AI skills to users',
    variants: {
      'proactive': {
        name: 'Proactive Suggestions',
        config: { style: 'proactive', timing: 'immediate' },
        weight: 0.5
      },
      'on-demand': {
        name: 'On-Demand Access',
        config: { style: 'on-demand', timing: 'user-initiated' },
        weight: 0.5
      }
    },
    metrics: ['skill_usage', 'completion_rate', 'perceived_value'],
    status: 'active'
  },
  'personalization-depth': {
    name: 'Personalization Depth',
    description: 'Test different levels of personalization',
    variants: {
      'minimal': {
        name: 'Minimal Personalization',
        config: { depth: 'minimal', factors: ['time', 'device'] },
        weight: 0.25
      },
      'moderate': {
        name: 'Moderate Personalization',
        config: { depth: 'moderate', factors: ['time', 'device', 'behavior'] },
        weight: 0.5
      },
      'deep': {
        name: 'Deep Personalization',
        config: { depth: 'deep', factors: ['time', 'device', 'behavior', 'social', 'context'] },
        weight: 0.25
      }
    },
    metrics: ['relevance_score', 'engagement_time', 'return_rate'],
    status: 'active'
  }
};

// Feature Flags
const FEATURE_FLAGS = {
  'voice-search': {
    name: 'Voice Search',
    description: 'Enable voice search functionality',
    enabled: true,
    rollout: 0.8, // 80% rollout
    conditions: ['device_support', 'user_consent']
  },
  'real-time-collaboration': {
    name: 'Real-time Collaboration',
    description: 'Enable real-time collaborative features',
    enabled: false,
    rollout: 0.1, // 10% rollout
    conditions: ['premium_user', 'group_member']
  },
  'advanced-ai-skills': {
    name: 'Advanced AI Skills',
    description: 'Enable advanced AI skill orchestration',
    enabled: true,
    rollout: 0.6, // 60% rollout
    conditions: ['active_user', 'skill_engagement']
  },
  'contextual-notifications': {
    name: 'Contextual Notifications',
    description: 'Smart, context-aware notifications',
    enabled: true,
    rollout: 0.9, // 90% rollout
    conditions: ['notification_permission']
  }
};

const ExperimentationEngine = ({ isVisible, onClose }) => {
  const [activeExperiments, setActiveExperiments] = useState({});
  const [featureFlags, setFeatureFlags] = useState(FEATURE_FLAGS);
  const [metrics, setMetrics] = useState({});
  const [userVariants, setUserVariants] = useState({});
  const [experimentResults, setExperimentResults] = useState({});
  
  const metricsCollectionRef = useRef({});
  const experimentStartTime = useRef(Date.now());
  
  const {
    state,
    trackInteraction,
    shouldShowFeature
  } = useContextAware();

  // Initialize user variants for experiments
  const initializeUserVariants = useCallback(() => {
    const variants = {};
    const userId = state.user.id;
    
    Object.entries(EXPERIMENTS).forEach(([experimentId, experiment]) => {
      if (experiment.status === 'active') {
        // Use deterministic assignment based on user ID
        const hash = hashString(userId + experimentId);
        const random = (hash % 1000) / 1000;
        
        let cumulativeWeight = 0;
        for (const [variantId, variant] of Object.entries(experiment.variants)) {
          cumulativeWeight += variant.weight;
          if (random <= cumulativeWeight) {
            variants[experimentId] = {
              variantId,
              config: variant.config,
              assignedAt: Date.now()
            };
            break;
          }
        }
      }
    });
    
    setUserVariants(variants);
    return variants;
  }, [state.user.id]);

  // Hash function for deterministic assignment
  const hashString = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  };

  // Check if user should see a feature flag
  const shouldShowFeatureFlag = useCallback((flagId) => {
    const flag = featureFlags[flagId];
    if (!flag || !flag.enabled) return false;
    
    // Check rollout percentage
    const userId = state.user.id;
    const hash = hashString(userId + flagId);
    const random = (hash % 1000) / 1000;
    
    if (random > flag.rollout) return false;
    
    // Check conditions
    return flag.conditions.every(condition => {
      switch (condition) {
        case 'device_support':
          return 'webkitSpeechRecognition' in window;
        case 'user_consent':
          return localStorage.getItem('voice_consent') === 'true';
        case 'premium_user':
          return state.user.preferences.tier === 'premium';
        case 'group_member':
          return state.session.currentView === 'groups';
        case 'active_user':
          return state.user.behavior.articlesRead > 5;
        case 'skill_engagement':
          return state.session.interactionHistory.some(i => i.type === 'skill_use');
        case 'notification_permission':
          return Notification.permission === 'granted';
        default:
          return true;
      }
    });
  }, [featureFlags, state]);

  // Get experiment variant configuration
  const getExperimentVariant = useCallback((experimentId) => {
    return userVariants[experimentId];
  }, [userVariants]);

  // Track experiment metrics
  const trackExperimentMetric = useCallback((experimentId, metric, value, context = {}) => {
    const variant = userVariants[experimentId];
    if (!variant) return;
    
    const metricKey = `${experimentId}_${variant.variantId}_${metric}`;
    const timestamp = Date.now();
    
    if (!metricsCollectionRef.current[metricKey]) {
      metricsCollectionRef.current[metricKey] = [];
    }
    
    metricsCollectionRef.current[metricKey].push({
      value,
      timestamp,
      context,
      sessionId: state.session.startTime.getTime(),
      userId: state.user.id
    });
    
    // Update real-time metrics
    setMetrics(prev => ({
      ...prev,
      [metricKey]: {
        latest: value,
        count: (prev[metricKey]?.count || 0) + 1,
        average: calculateAverage(metricsCollectionRef.current[metricKey]),
        trend: calculateTrend(metricsCollectionRef.current[metricKey])
      }
    }));
    
    trackInteraction('experiment_metric', experimentId, {
      variant: variant.variantId,
      metric,
      value,
      context
    });
  }, [userVariants, state, trackInteraction]);

  // Calculate metric average
  const calculateAverage = (dataPoints) => {
    if (!dataPoints.length) return 0;
    const sum = dataPoints.reduce((acc, point) => acc + point.value, 0);
    return sum / dataPoints.length;
  };

  // Calculate metric trend
  const calculateTrend = (dataPoints) => {
    if (dataPoints.length < 2) return 0;
    const recent = dataPoints.slice(-5);
    const older = dataPoints.slice(-10, -5);
    
    if (older.length === 0) return 0;
    
    const recentAvg = recent.reduce((acc, p) => acc + p.value, 0) / recent.length;
    const olderAvg = older.reduce((acc, p) => acc + p.value, 0) / older.length;
    
    return ((recentAvg - olderAvg) / olderAvg) * 100;
  };

  // Analyze experiment results
  const analyzeExperimentResults = useCallback(() => {
    const results = {};
    
    Object.entries(EXPERIMENTS).forEach(([experimentId, experiment]) => {
      const variantResults = {};
      
      Object.keys(experiment.variants).forEach(variantId => {
        const variantMetrics = {};
        
        experiment.metrics.forEach(metric => {
          const metricKey = `${experimentId}_${variantId}_${metric}`;
          const metricData = metricsCollectionRef.current[metricKey] || [];
          
          variantMetrics[metric] = {
            sampleSize: metricData.length,
            average: calculateAverage(metricData),
            trend: calculateTrend(metricData),
            confidence: calculateConfidence(metricData)
          };
        });
        
        variantResults[variantId] = variantMetrics;
      });
      
      results[experimentId] = {
        ...experiment,
        results: variantResults,
        winner: determineWinner(variantResults, experiment.metrics),
        significance: calculateSignificance(variantResults)
      };
    });
    
    setExperimentResults(results);
  }, []);

  // Calculate statistical confidence
  const calculateConfidence = (dataPoints) => {
    if (dataPoints.length < 10) return 'low';
    if (dataPoints.length < 50) return 'medium';
    return 'high';
  };

  // Determine experiment winner
  const determineWinner = (variantResults, metrics) => {
    const scores = {};
    
    Object.entries(variantResults).forEach(([variantId, results]) => {
      scores[variantId] = metrics.reduce((score, metric) => {
        const metricResult = results[metric];
        if (!metricResult || metricResult.sampleSize < 5) return score;
        
        // Weight metrics (engagement and completion are more important)
        const weight = metric.includes('engagement') || metric.includes('completion') ? 2 : 1;
        return score + (metricResult.average * weight);
      }, 0);
    });
    
    const winner = Object.entries(scores).reduce((best, [variantId, score]) => {
      return score > best.score ? { variantId, score } : best;
    }, { variantId: null, score: -1 });
    
    return winner.variantId;
  };

  // Calculate statistical significance
  const calculateSignificance = (variantResults) => {
    const variants = Object.keys(variantResults);
    if (variants.length < 2) return 'insufficient-data';
    
    const sampleSizes = variants.map(v => 
      Object.values(variantResults[v]).reduce((sum, metric) => sum + metric.sampleSize, 0)
    );
    
    const minSampleSize = Math.min(...sampleSizes);
    
    if (minSampleSize < 30) return 'insufficient-data';
    if (minSampleSize < 100) return 'low';
    if (minSampleSize < 500) return 'medium';
    return 'high';
  };

  // Initialize experiments on mount
  useEffect(() => {
    const variants = initializeUserVariants();
    setActiveExperiments(EXPERIMENTS);
    
    // Start metrics collection
    const interval = setInterval(() => {
      analyzeExperimentResults();
    }, 30000); // Analyze every 30 seconds
    
    return () => clearInterval(interval);
  }, [initializeUserVariants, analyzeExperimentResults]);

  // Expose experiment functions globally for use in other components
  useEffect(() => {
    window.experimentationEngine = {
      shouldShowFeatureFlag,
      getExperimentVariant,
      trackExperimentMetric,
      userVariants
    };
    
    return () => {
      delete window.experimentationEngine;
    };
  }, [shouldShowFeatureFlag, getExperimentVariant, trackExperimentMetric, userVariants]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-6 h-6 text-blue-600" />
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Experimentation Engine
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Real-time A/B testing and feature flag management
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex h-[70vh]">
          {/* Experiments Panel */}
          <div className="w-1/2 p-6 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Active Experiments
            </h3>
            
            <div className="space-y-4">
              {Object.entries(experimentResults).map(([experimentId, experiment]) => {
                const userVariant = userVariants[experimentId];
                return (
                  <div key={experimentId} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {experiment.name}
                      </h4>
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${
                          experiment.significance === 'high' ? 'bg-green-500' :
                          experiment.significance === 'medium' ? 'bg-yellow-500' :
                          experiment.significance === 'low' ? 'bg-orange-500' : 'bg-gray-400'
                        }`} />
                        <span className="text-xs text-gray-500">
                          {experiment.significance}
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {experiment.description}
                    </p>
                    
                    {userVariant && (
                      <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg mb-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Target className="w-4 h-4 text-blue-600" />
                          <span className="text-sm font-medium text-blue-800 dark:text-blue-400">
                            Your Variant: {experiment.variants[userVariant.variantId]?.name}
                          </span>
                        </div>
                        <p className="text-xs text-blue-700 dark:text-blue-300">
                          Assigned {new Date(userVariant.assignedAt).toLocaleDateString()}
                        </p>
                      </div>
                    )}
                    
                    <div className="space-y-2">
                      {Object.entries(experiment.results || {}).map(([variantId, results]) => {
                        const isWinner = experiment.winner === variantId;
                        const isUserVariant = userVariant?.variantId === variantId;
                        
                        return (
                          <div key={variantId} className={`p-2 rounded border ${
                            isWinner ? 'border-green-300 bg-green-50 dark:bg-green-900/20' :
                            isUserVariant ? 'border-blue-300 bg-blue-50 dark:bg-blue-900/20' :
                            'border-gray-200 dark:border-gray-700'
                          }`}>
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium">
                                {experiment.variants[variantId]?.name}
                                {isWinner && ' 🏆'}
                                {isUserVariant && ' (You)'}
                              </span>
                              <div className="flex gap-2 text-xs">
                                {experiment.metrics.map(metric => {
                                  const metricData = results[metric];
                                  return (
                                    <span key={metric} className="text-gray-500">
                                      {metric}: {metricData?.average?.toFixed(2) || 'N/A'}
                                    </span>
                                  );
                                })}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Feature Flags Panel */}
          <div className="w-1/2 p-6 overflow-y-auto">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Feature Flags
            </h3>
            
            <div className="space-y-4">
              {Object.entries(featureFlags).map(([flagId, flag]) => {
                const isEnabled = shouldShowFeatureFlag(flagId);
                
                return (
                  <div key={flagId} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {flag.name}
                      </h4>
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${
                          isEnabled ? 'bg-green-500' : 'bg-gray-400'
                        }`} />
                        <span className="text-xs text-gray-500">
                          {isEnabled ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {flag.description}
                    </p>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">
                        Rollout: {Math.round(flag.rollout * 100)}%
                      </span>
                      <div className="flex gap-1">
                        {flag.conditions.map((condition, index) => (
                          <span key={index} className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded text-xs">
                            {condition}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Real-time Metrics */}
            <div className="mt-6">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Real-time Metrics
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Users className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium">Active Users</span>
                  </div>
                  <div className="text-2xl font-bold text-blue-700 dark:text-blue-400">
                    {Object.keys(userVariants).length}
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="w-4 h-4 text-green-600" />
                    <span className="text-sm font-medium">Experiments</span>
                  </div>
                  <div className="text-2xl font-bold text-green-700 dark:text-green-400">
                    {Object.keys(activeExperiments).length}
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-4 h-4 text-orange-600" />
                    <span className="text-sm font-medium">Features</span>
                  </div>
                  <div className="text-2xl font-bold text-orange-700 dark:text-orange-400">
                    {Object.values(featureFlags).filter(f => f.enabled).length}
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Clock className="w-4 h-4 text-purple-600" />
                    <span className="text-sm font-medium">Session Time</span>
                  </div>
                  <div className="text-2xl font-bold text-purple-700 dark:text-purple-400">
                    {Math.round((Date.now() - experimentStartTime.current) / 60000)}m
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExperimentationEngine;