import React, { useState, useEffect, useCallback } from 'react';
import { Brain, Zap, TrendingUp, FileText, Share2, BookOpen, MessageSquare, BarChart3, Lightbulb, Clock } from 'lucide-react';
import { useContextAware } from '../contexts/ContextProvider';

// AI Skill Definitions
const AI_SKILLS = {
  summarize: {
    name: 'Summarize',
    icon: FileText,
    description: 'Create concise summaries of articles or topics',
    triggers: ['long_read', 'multiple_articles', 'time_constraint'],
    contextRequirements: ['article_content', 'reading_time'],
    confidence: 0.9
  },
  compare: {
    name: 'Compare',
    icon: BarChart3,
    description: 'Compare different viewpoints, articles, or topics',
    triggers: ['multiple_searches', 'conflicting_info', 'research_mode'],
    contextRequirements: ['search_history', 'multiple_sources'],
    confidence: 0.8
  },
  explain: {
    name: 'Explain',
    icon: Lightbulb,
    description: 'Break down complex topics into understandable explanations',
    triggers: ['complex_topic', 'technical_content', 'learning_mode'],
    contextRequirements: ['user_expertise', 'topic_complexity'],
    confidence: 0.85
  },
  draft: {
    name: 'Draft',
    icon: Share2,
    description: 'Create social media posts, emails, or summaries',
    triggers: ['sharing_intent', 'peak_hours', 'engagement_mode'],
    contextRequirements: ['platform_preference', 'audience_type'],
    confidence: 0.75
  },
  trend: {
    name: 'Trend Analysis',
    icon: TrendingUp,
    description: 'Analyze trends and predict future developments',
    triggers: ['pattern_recognition', 'future_planning', 'strategic_thinking'],
    contextRequirements: ['historical_data', 'trend_patterns'],
    confidence: 0.7
  },
  discuss: {
    name: 'Discuss',
    icon: MessageSquare,
    description: 'Facilitate discussions and generate talking points',
    triggers: ['social_engagement', 'group_activity', 'debate_mode'],
    contextRequirements: ['social_context', 'discussion_topic'],
    confidence: 0.8
  },
  research: {
    name: 'Research',
    icon: BookOpen,
    description: 'Deep dive into topics with comprehensive analysis',
    triggers: ['academic_mode', 'professional_research', 'detailed_inquiry'],
    contextRequirements: ['research_depth', 'source_credibility'],
    confidence: 0.9
  }
};

const SkillOrchestrator = ({ isVisible, currentContext, onSkillSelect, onClose }) => {
  const [availableSkills, setAvailableSkills] = useState([]);
  const [selectedSkill, setSelectedSkill] = useState(null);
  const [skillParameters, setSkillParameters] = useState({});
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);
  
  const {
    state,
    trackInteraction,
    getSuggestedSkills,
    shouldShowFeature
  } = useContextAware();

  // Analyze context and determine available skills
  const analyzeContextAndSkills = useCallback(() => {
    const contextualSkills = [];
    const currentTime = new Date().getHours();
    const { user, session, environment } = state;

    // Analyze each skill for relevance
    Object.entries(AI_SKILLS).forEach(([skillId, skill]) => {
      let relevanceScore = 0;
      let triggers = [];

      // Check trigger conditions
      skill.triggers.forEach(trigger => {
        switch (trigger) {
          case 'long_read':
            if (user.behavior.articlesRead > 2) {
              relevanceScore += 0.3;
              triggers.push('Multiple articles read');
            }
            break;
          
          case 'multiple_articles':
            if (session.readArticles.length > 1) {
              relevanceScore += 0.4;
              triggers.push('Multiple articles in session');
            }
            break;
          
          case 'time_constraint':
            if (environment.timeOfDay === 'morning' || user.behavior.scrollPattern === 'fast') {
              relevanceScore += 0.2;
              triggers.push('Time-sensitive reading');
            }
            break;
          
          case 'multiple_searches':
            if (session.searchHistory.length > 2) {
              relevanceScore += 0.4;
              triggers.push('Research pattern detected');
            }
            break;
          
          case 'sharing_intent':
            if (session.sharedArticles.length > 0 || (currentTime >= 9 && currentTime <= 17)) {
              relevanceScore += 0.3;
              triggers.push('Sharing activity or peak hours');
            }
            break;
          
          case 'peak_hours':
            if ((currentTime >= 8 && currentTime <= 10) || (currentTime >= 17 && currentTime <= 19)) {
              relevanceScore += 0.2;
              triggers.push('Peak engagement hours');
            }
            break;
          
          case 'social_engagement':
            if (shouldShowFeature('social-recommendations')) {
              relevanceScore += 0.3;
              triggers.push('Social features active');
            }
            break;
          
          case 'complex_topic':
            if (currentContext?.complexity === 'high' || user.behavior.searchQueries.some(q => q.length > 20)) {
              relevanceScore += 0.4;
              triggers.push('Complex topic detected');
            }
            break;
          
          case 'pattern_recognition':
            if (session.searchHistory.length > 3) {
              relevanceScore += 0.3;
              triggers.push('Pattern in search behavior');
            }
            break;
        }
      });

      // Add contextual bonus
      if (currentContext) {
        if (currentContext.type === 'article' && skillId === 'summarize') relevanceScore += 0.2;
        if (currentContext.type === 'search' && skillId === 'research') relevanceScore += 0.2;
        if (currentContext.type === 'social' && skillId === 'draft') relevanceScore += 0.2;
      }

      // Only include skills with sufficient relevance
      if (relevanceScore > 0.2) {
        contextualSkills.push({
          ...skill,
          id: skillId,
          relevanceScore,
          triggers,
          confidence: Math.min(skill.confidence + relevanceScore, 1.0)
        });
      }
    });

    // Sort by relevance and confidence
    contextualSkills.sort((a, b) => (b.relevanceScore + b.confidence) - (a.relevanceScore + a.confidence));
    
    setAvailableSkills(contextualSkills.slice(0, 6)); // Show top 6 skills
  }, [state, currentContext, shouldShowFeature]);

  // Update skills when context changes
  useEffect(() => {
    if (isVisible) {
      analyzeContextAndSkills();
      trackInteraction('skill_orchestrator_open', 'ai_skills');
    }
  }, [isVisible, analyzeContextAndSkills, trackInteraction]);

  // Handle skill selection
  const handleSkillSelect = useCallback((skill) => {
    setSelectedSkill(skill);
    
    // Generate default parameters based on context
    const defaultParams = generateSkillParameters(skill);
    setSkillParameters(defaultParams);
    
    trackInteraction('skill_select', skill.id, {
      confidence: skill.confidence,
      triggers: skill.triggers
    });
  }, [trackInteraction]);

  // Generate skill parameters based on context
  const generateSkillParameters = useCallback((skill) => {
    const params = {};
    const { user, session, environment } = state;

    switch (skill.id) {
      case 'summarize':
        params.length = user.behavior.scrollPattern === 'fast' ? 'brief' : 'detailed';
        params.style = environment.timeOfDay === 'morning' ? 'bullet-points' : 'paragraph';
        params.focus = currentContext?.topic || 'main-points';
        break;
      
      case 'compare':
        params.sources = session.searchHistory.slice(-3).map(s => s.query);
        params.perspective = 'balanced';
        params.format = 'table';
        break;
      
      case 'explain':
        params.level = user.preferences.readingSpeed === 'fast' ? 'intermediate' : 'beginner';
        params.examples = true;
        params.analogies = environment.deviceType === 'mobile';
        break;
      
      case 'draft':
        params.platform = 'twitter'; // Default, can be changed
        params.tone = environment.timeOfDay === 'morning' ? 'professional' : 'casual';
        params.length = environment.deviceType === 'mobile' ? 'short' : 'medium';
        break;
      
      case 'trend':
        params.timeframe = '6-months';
        params.includeData = true;
        params.predictions = true;
        break;
      
      case 'discuss':
        params.perspective = 'multiple';
        params.questions = 3;
        params.format = 'structured';
        break;
      
      case 'research':
        params.depth = 'comprehensive';
        params.sources = 'academic';
        params.citations = true;
        break;
    }

    return params;
  }, [state, currentContext]);

  // Execute selected skill
  const executeSkill = useCallback(async () => {
    if (!selectedSkill) return;
    
    setIsExecuting(true);
    trackInteraction('skill_execute', selectedSkill.id, skillParameters);
    
    try {
      // Simulate API call to skill execution endpoint
      const response = await fetch('/api/ai/execute-skill', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          skill: selectedSkill.id,
          parameters: skillParameters,
          context: {
            userContext: state.user,
            sessionContext: state.session,
            environmentContext: state.environment,
            currentContext
          }
        })
      });
      
      const result = await response.json();
      setExecutionResult(result);
      
      // Notify parent component
      if (onSkillSelect) {
        onSkillSelect({
          skill: selectedSkill,
          parameters: skillParameters,
          result
        });
      }
    } catch (error) {
      console.error('Skill execution failed:', error);
      setExecutionResult({
        error: 'Failed to execute skill. Please try again.',
        success: false
      });
    } finally {
      setIsExecuting(false);
    }
  }, [selectedSkill, skillParameters, state, currentContext, onSkillSelect, trackInteraction]);

  // Reset state when closing
  const handleClose = useCallback(() => {
    setSelectedSkill(null);
    setSkillParameters({});
    setExecutionResult(null);
    if (onClose) onClose();
  }, [onClose]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="w-6 h-6 text-purple-600" />
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  AI Skill Orchestrator
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Context-aware AI skills tailored to your current activity
                </p>
              </div>
            </div>
            <button
              onClick={handleClose}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex h-[60vh]">
          {/* Skills List */}
          <div className="w-1/2 p-6 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Available Skills
            </h3>
            <div className="space-y-3">
              {availableSkills.map((skill) => {
                const IconComponent = skill.icon;
                return (
                  <button
                    key={skill.id}
                    onClick={() => handleSkillSelect(skill)}
                    className={`w-full p-4 rounded-xl border-2 transition-all text-left ${
                      selectedSkill?.id === skill.id
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-purple-300 hover:bg-gray-50 dark:hover:bg-gray-800'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`p-2 rounded-lg ${
                        selectedSkill?.id === skill.id
                          ? 'bg-purple-100 text-purple-600 dark:bg-purple-800 dark:text-purple-400'
                          : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                      }`}>
                        <IconComponent className="w-5 h-5" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="font-medium text-gray-900 dark:text-white">
                            {skill.name}
                          </h4>
                          <div className="flex items-center gap-1">
                            <div className={`w-2 h-2 rounded-full ${
                              skill.confidence > 0.8 ? 'bg-green-500' :
                              skill.confidence > 0.6 ? 'bg-yellow-500' : 'bg-gray-400'
                            }`} />
                            <span className="text-xs text-gray-500">
                              {Math.round(skill.confidence * 100)}%
                            </span>
                          </div>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                          {skill.description}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {skill.triggers.slice(0, 2).map((trigger, index) => (
                            <span
                              key={index}
                              className="text-xs px-2 py-1 bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 rounded-full"
                            >
                              {trigger}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Skill Configuration & Execution */}
          <div className="w-1/2 p-6 overflow-y-auto">
            {selectedSkill ? (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    Configure {selectedSkill.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedSkill.description}
                  </p>
                </div>

                {/* Skill Parameters */}
                <div className="space-y-4">
                  {Object.entries(skillParameters).map(([key, value]) => (
                    <div key={key}>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                      </label>
                      {typeof value === 'boolean' ? (
                        <input
                          type="checkbox"
                          checked={value}
                          onChange={(e) => setSkillParameters(prev => ({
                            ...prev,
                            [key]: e.target.checked
                          }))}
                          className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                        />
                      ) : Array.isArray(value) ? (
                        <select
                          value={value[0] || ''}
                          onChange={(e) => setSkillParameters(prev => ({
                            ...prev,
                            [key]: [e.target.value]
                          }))}
                          className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
                        >
                          {value.map((option, index) => (
                            <option key={index} value={option}>{option}</option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="text"
                          value={value}
                          onChange={(e) => setSkillParameters(prev => ({
                            ...prev,
                            [key]: e.target.value
                          }))}
                          className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
                        />
                      )}
                    </div>
                  ))}
                </div>

                {/* Execution Button */}
                <button
                  onClick={executeSkill}
                  disabled={isExecuting}
                  className="w-full py-3 px-4 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {isExecuting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Executing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Execute {selectedSkill.name}
                    </>
                  )}
                </button>

                {/* Execution Result */}
                {executionResult && (
                  <div className={`p-4 rounded-lg ${
                    executionResult.error
                      ? 'bg-red-50 border border-red-200 dark:bg-red-900/20 dark:border-red-800'
                      : 'bg-green-50 border border-green-200 dark:bg-green-900/20 dark:border-green-800'
                  }`}>
                    <h4 className={`font-medium mb-2 ${
                      executionResult.error ? 'text-red-800 dark:text-red-400' : 'text-green-800 dark:text-green-400'
                    }`}>
                      {executionResult.error ? 'Execution Failed' : 'Execution Complete'}
                    </h4>
                    <p className={`text-sm ${
                      executionResult.error ? 'text-red-700 dark:text-red-300' : 'text-green-700 dark:text-green-300'
                    }`}>
                      {executionResult.error || executionResult.message || 'Skill executed successfully'}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-center">
                <div>
                  <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    Select an AI Skill
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Choose a skill from the list to configure and execute it
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SkillOrchestrator;