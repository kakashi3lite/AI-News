import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react';

// Context-Aware State Interface
interface ContextState {
  // User Context
  user: {
    id: string;
    preferences: {
      readingSpeed: 'slow' | 'medium' | 'fast';
      preferredTopics: string[];
      readingTime: 'morning' | 'afternoon' | 'evening' | 'night';
      deviceType: 'mobile' | 'tablet' | 'desktop';
    };
    behavior: {
      sessionDuration: number;
      articlesRead: number;
      searchQueries: string[];
      lastActivity: Date;
      typingSpeed: number;
      scrollPattern: 'fast' | 'slow' | 'detailed';
    };
  };
  
  // Environmental Context
  environment: {
    timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
    deviceType: 'mobile' | 'tablet' | 'desktop';
    networkQuality: 'slow' | 'medium' | 'fast';
    batteryLevel?: number;
    isOnline: boolean;
    viewport: { width: number; height: number };
  };
  
  // Session Context
  session: {
    startTime: Date;
    currentView: 'news' | 'social' | 'groups' | 'profile';
    searchHistory: Array<{ query: string; timestamp: Date; results: number }>;
    readArticles: string[];
    sharedArticles: string[];
    bookmarkedArticles: string[];
    interactionHistory: Array<{
      type: 'click' | 'scroll' | 'search' | 'share' | 'bookmark';
      target: string;
      timestamp: Date;
      context: any;
    }>;
  };
  
  // AI Context
  ai: {
    recommendationEngine: 'collaborative' | 'content' | 'hybrid';
    personalizedPrompts: string[];
    suggestedSkills: Array<{
      name: string;
      trigger: string;
      confidence: number;
    }>;
    contextualHints: Array<{
      message: string;
      type: 'tip' | 'suggestion' | 'warning';
      priority: number;
    }>;
  };
}

// Action Types
type ContextAction = 
  | { type: 'UPDATE_USER_BEHAVIOR'; payload: Partial<ContextState['user']['behavior']> }
  | { type: 'UPDATE_ENVIRONMENT'; payload: Partial<ContextState['environment']> }
  | { type: 'ADD_INTERACTION'; payload: ContextState['session']['interactionHistory'][0] }
  | { type: 'ADD_SEARCH'; payload: { query: string; results: number } }
  | { type: 'SET_VIEW'; payload: ContextState['session']['currentView'] }
  | { type: 'UPDATE_AI_CONTEXT'; payload: Partial<ContextState['ai']> }
  | { type: 'MARK_ARTICLE_READ'; payload: string }
  | { type: 'UPDATE_TYPING_SPEED'; payload: number }
  | { type: 'UPDATE_SCROLL_PATTERN'; payload: 'fast' | 'slow' | 'detailed' };

// Initial State
const initialState: ContextState = {
  user: {
    id: 'user-' + Math.random().toString(36).substr(2, 9),
    preferences: {
      readingSpeed: 'medium',
      preferredTopics: [],
      readingTime: 'morning',
      deviceType: 'desktop'
    },
    behavior: {
      sessionDuration: 0,
      articlesRead: 0,
      searchQueries: [],
      lastActivity: new Date(),
      typingSpeed: 0,
      scrollPattern: 'medium'
    }
  },
  environment: {
    timeOfDay: 'morning',
    deviceType: 'desktop',
    networkQuality: 'fast',
    isOnline: navigator.onLine,
    viewport: { width: window.innerWidth, height: window.innerHeight }
  },
  session: {
    startTime: new Date(),
    currentView: 'news',
    searchHistory: [],
    readArticles: [],
    sharedArticles: [],
    bookmarkedArticles: [],
    interactionHistory: []
  },
  ai: {
    recommendationEngine: 'hybrid',
    personalizedPrompts: [],
    suggestedSkills: [],
    contextualHints: []
  }
};

// Context Reducer
function contextReducer(state: ContextState, action: ContextAction): ContextState {
  switch (action.type) {
    case 'UPDATE_USER_BEHAVIOR':
      return {
        ...state,
        user: {
          ...state.user,
          behavior: { ...state.user.behavior, ...action.payload, lastActivity: new Date() }
        }
      };
    
    case 'UPDATE_ENVIRONMENT':
      return {
        ...state,
        environment: { ...state.environment, ...action.payload }
      };
    
    case 'ADD_INTERACTION':
      return {
        ...state,
        session: {
          ...state.session,
          interactionHistory: [...state.session.interactionHistory.slice(-49), action.payload]
        }
      };
    
    case 'ADD_SEARCH':
      const searchEntry = {
        query: action.payload.query,
        timestamp: new Date(),
        results: action.payload.results
      };
      return {
        ...state,
        session: {
          ...state.session,
          searchHistory: [...state.session.searchHistory.slice(-19), searchEntry]
        },
        user: {
          ...state.user,
          behavior: {
            ...state.user.behavior,
            searchQueries: [...state.user.behavior.searchQueries.slice(-9), action.payload.query]
          }
        }
      };
    
    case 'SET_VIEW':
      return {
        ...state,
        session: { ...state.session, currentView: action.payload }
      };
    
    case 'UPDATE_AI_CONTEXT':
      return {
        ...state,
        ai: { ...state.ai, ...action.payload }
      };
    
    case 'MARK_ARTICLE_READ':
      return {
        ...state,
        session: {
          ...state.session,
          readArticles: [...state.session.readArticles, action.payload]
        },
        user: {
          ...state.user,
          behavior: {
            ...state.user.behavior,
            articlesRead: state.user.behavior.articlesRead + 1
          }
        }
      };
    
    case 'UPDATE_TYPING_SPEED':
      return {
        ...state,
        user: {
          ...state.user,
          behavior: { ...state.user.behavior, typingSpeed: action.payload }
        }
      };
    
    case 'UPDATE_SCROLL_PATTERN':
      return {
        ...state,
        user: {
          ...state.user,
          behavior: { ...state.user.behavior, scrollPattern: action.payload }
        }
      };
    
    default:
      return state;
  }
}

// Context Creation
const ContextAwareContext = createContext<{
  state: ContextState;
  dispatch: React.Dispatch<ContextAction>;
  trackInteraction: (type: string, target: string, context?: any) => void;
  getPersonalizedPrompts: () => string[];
  getSuggestedSkills: () => Array<{ name: string; trigger: string; confidence: number }>;
  getContextualHints: () => Array<{ message: string; type: string; priority: number }>;
  shouldShowFeature: (feature: string) => boolean;
} | null>(null);

// Context Provider Component
export const ContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(contextReducer, initialState);

  // Track user interactions
  const trackInteraction = useCallback((type: string, target: string, context: any = {}) => {
    dispatch({
      type: 'ADD_INTERACTION',
      payload: {
        type: type as any,
        target,
        timestamp: new Date(),
        context
      }
    });
  }, []);

  // Get personalized prompts based on context
  const getPersonalizedPrompts = useCallback(() => {
    const { user, environment, session } = state;
    const prompts = [];

    // Time-based prompts
    if (environment.timeOfDay === 'morning') {
      prompts.push('Good morning! Here\'s your daily AI news digest');
    } else if (environment.timeOfDay === 'evening') {
      prompts.push('Catch up on today\'s most important AI developments');
    }

    // Behavior-based prompts
    if (session.readArticles.length > 3) {
      prompts.push('Ready to share your insights? Draft a summary tweet');
    }

    if (user.behavior.searchQueries.length > 2) {
      prompts.push('Create a custom alert for your research topics');
    }

    return prompts;
  }, [state]);

  // Get suggested AI skills based on context
  const getSuggestedSkills = useCallback(() => {
    const { user, session } = state;
    const skills = [];

    // Reading pattern analysis
    if (user.behavior.articlesRead > 2) {
      skills.push({
        name: 'Summarize Articles',
        trigger: 'After reading multiple articles',
        confidence: 0.8
      });
    }

    // Search pattern analysis
    if (session.searchHistory.length > 1) {
      skills.push({
        name: 'Compare Topics',
        trigger: 'Based on your search history',
        confidence: 0.7
      });
    }

    // Time-based skills
    if (state.environment.timeOfDay === 'evening') {
      skills.push({
        name: 'Daily Digest',
        trigger: 'End of day summary',
        confidence: 0.9
      });
    }

    return skills;
  }, [state]);

  // Get contextual hints
  const getContextualHints = useCallback(() => {
    const { environment, user } = state;
    const hints = [];

    // Device-specific hints
    if (environment.deviceType === 'mobile') {
      hints.push({
        message: 'Swipe right for quick actions',
        type: 'tip',
        priority: 1
      });
    }

    // Network-specific hints
    if (environment.networkQuality === 'slow') {
      hints.push({
        message: 'Slow connection detected. Loading optimized content.',
        type: 'warning',
        priority: 2
      });
    }

    // Behavior-specific hints
    if (user.behavior.typingSpeed > 100) {
      hints.push({
        message: 'Try voice search for faster queries',
        type: 'suggestion',
        priority: 1
      });
    }

    return hints.sort((a, b) => b.priority - a.priority);
  }, [state]);

  // Determine if a feature should be shown based on context
  const shouldShowFeature = useCallback((feature: string) => {
    const { user, environment, session } = state;

    switch (feature) {
      case 'social-recommendations':
        return session.readArticles.length > 1;
      
      case 'voice-search':
        return environment.deviceType === 'mobile' || user.behavior.typingSpeed > 80;
      
      case 'reading-mode':
        return user.behavior.articlesRead > 0 && environment.timeOfDay === 'evening';
      
      case 'collaboration-tools':
        return session.sharedArticles.length > 0 || user.behavior.searchQueries.length > 2;
      
      default:
        return true;
    }
  }, [state]);

  // Environmental context tracking
  useEffect(() => {
    // Time of day detection
    const updateTimeOfDay = () => {
      const hour = new Date().getHours();
      let timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
      
      if (hour >= 5 && hour < 12) timeOfDay = 'morning';
      else if (hour >= 12 && hour < 17) timeOfDay = 'afternoon';
      else if (hour >= 17 && hour < 22) timeOfDay = 'evening';
      else timeOfDay = 'night';
      
      dispatch({ type: 'UPDATE_ENVIRONMENT', payload: { timeOfDay } });
    };

    // Device type detection
    const updateDeviceType = () => {
      const width = window.innerWidth;
      let deviceType: 'mobile' | 'tablet' | 'desktop';
      
      if (width < 768) deviceType = 'mobile';
      else if (width < 1024) deviceType = 'tablet';
      else deviceType = 'desktop';
      
      dispatch({ type: 'UPDATE_ENVIRONMENT', payload: { deviceType } });
    };

    // Network quality detection
    const updateNetworkQuality = () => {
      if ('connection' in navigator) {
        const connection = (navigator as any).connection;
        let networkQuality: 'slow' | 'medium' | 'fast';
        
        if (connection.effectiveType === '4g') networkQuality = 'fast';
        else if (connection.effectiveType === '3g') networkQuality = 'medium';
        else networkQuality = 'slow';
        
        dispatch({ type: 'UPDATE_ENVIRONMENT', payload: { networkQuality } });
      }
    };

    // Online status
    const updateOnlineStatus = () => {
      dispatch({ type: 'UPDATE_ENVIRONMENT', payload: { isOnline: navigator.onLine } });
    };

    // Viewport tracking
    const updateViewport = () => {
      dispatch({
        type: 'UPDATE_ENVIRONMENT',
        payload: {
          viewport: { width: window.innerWidth, height: window.innerHeight }
        }
      });
    };

    // Session duration tracking
    const sessionTimer = setInterval(() => {
      dispatch({
        type: 'UPDATE_USER_BEHAVIOR',
        payload: {
          sessionDuration: Date.now() - state.session.startTime.getTime()
        }
      });
    }, 30000); // Update every 30 seconds

    // Event listeners
    updateTimeOfDay();
    updateDeviceType();
    updateNetworkQuality();
    updateOnlineStatus();
    updateViewport();

    window.addEventListener('resize', updateViewport);
    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);

    // Time updates
    const timeInterval = setInterval(updateTimeOfDay, 60000); // Check every minute

    return () => {
      clearInterval(sessionTimer);
      clearInterval(timeInterval);
      window.removeEventListener('resize', updateViewport);
      window.removeEventListener('online', updateOnlineStatus);
      window.removeEventListener('offline', updateOnlineStatus);
    };
  }, [state.session.startTime]);

  const value = {
    state,
    dispatch,
    trackInteraction,
    getPersonalizedPrompts,
    getSuggestedSkills,
    getContextualHints,
    shouldShowFeature
  };

  return (
    <ContextAwareContext.Provider value={value}>
      {children}
    </ContextAwareContext.Provider>
  );
};

// Custom hook to use context
export const useContextAware = () => {
  const context = useContext(ContextAwareContext);
  if (!context) {
    throw new Error('useContextAware must be used within a ContextProvider');
  }
  return context;
};

// Typing speed tracker hook
export const useTypingTracker = () => {
  const { dispatch } = useContextAware();
  
  const trackTyping = useCallback((inputElement: HTMLInputElement) => {
    let keyCount = 0;
    let startTime = Date.now();
    
    const handleKeyPress = () => {
      keyCount++;
      const elapsed = Date.now() - startTime;
      
      if (elapsed > 5000) { // Calculate WPM after 5 seconds
        const wpm = Math.round((keyCount / 5) * 60 / 5); // Approximate WPM
        dispatch({ type: 'UPDATE_TYPING_SPEED', payload: wpm });
        keyCount = 0;
        startTime = Date.now();
      }
    };
    
    inputElement.addEventListener('keypress', handleKeyPress);
    
    return () => {
      inputElement.removeEventListener('keypress', handleKeyPress);
    };
  }, [dispatch]);
  
  return trackTyping;
};

// Scroll pattern tracker hook
export const useScrollTracker = () => {
  const { dispatch } = useContextAware();
  
  useEffect(() => {
    let scrollCount = 0;
    let scrollDistance = 0;
    let lastScrollTime = Date.now();
    let lastScrollY = window.scrollY;
    
    const handleScroll = () => {
      const currentTime = Date.now();
      const currentScrollY = window.scrollY;
      const distance = Math.abs(currentScrollY - lastScrollY);
      const timeDiff = currentTime - lastScrollTime;
      
      scrollCount++;
      scrollDistance += distance;
      
      if (scrollCount > 10) {
        const avgSpeed = scrollDistance / timeDiff;
        let pattern: 'fast' | 'slow' | 'detailed';
        
        if (avgSpeed > 2) pattern = 'fast';
        else if (avgSpeed < 0.5) pattern = 'detailed';
        else pattern = 'slow';
        
        dispatch({ type: 'UPDATE_SCROLL_PATTERN', payload: pattern });
        
        // Reset counters
        scrollCount = 0;
        scrollDistance = 0;
      }
      
      lastScrollTime = currentTime;
      lastScrollY = currentScrollY;
    };
    
    window.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [dispatch]);
};

export default ContextProvider;