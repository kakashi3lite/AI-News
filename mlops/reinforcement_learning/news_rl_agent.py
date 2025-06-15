#!/usr/bin/env python3
"""
Dr. NewsForge's Reinforcement Learning News Agent

Advanced reinforcement learning system for personalized news recommendation,
user engagement optimization, and adaptive content curation.
Implements multi-agent RL, contextual bandits, and deep Q-learning
for real-time news personalization and engagement maximization.

Features:
- Deep Q-Network (DQN) for news recommendation
- Multi-Armed Bandits for A/B testing
- Actor-Critic methods for content optimization
- Contextual bandits for personalization
- Multi-agent reinforcement learning
- Real-time reward optimization
- User behavior modeling
- Content diversity optimization
- Exploration vs exploitation strategies
- Federated reinforcement learning

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uuid
import pickle
import hashlib
from pathlib import Path
import math
import random
from copy import deepcopy
import threading
from queue import Queue, PriorityQueue

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Categorical, Normal

# Reinforcement learning libraries
try:
    import gym
    from gym import spaces
    import stable_baselines3 as sb3
    from stable_baselines3 import DQN, PPO, A2C, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    RL_LIBS_AVAILABLE = True
except ImportError:
    RL_LIBS_AVAILABLE = False
    logging.warning("RL libraries not available, using custom implementation")

# Transformers and NLP
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, GPT2Model, T5Model
)
from sentence_transformers import SentenceTransformer

# Scientific computing
from scipy.optimize import minimize
from scipy.stats import beta, gamma
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Bandits and optimization
try:
    import vowpalwabbit as vw
    VW_AVAILABLE = True
except ImportError:
    VW_AVAILABLE = False
    logging.warning("VowpalWabbit not available, using custom bandits")

# Graph neural networks
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False
    logging.warning("PyTorch Geometric not available")

# Monitoring and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# MLOps and tracking
import mlflow
import wandb
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Networking
import requests
from flask import Flask, request, jsonify
import redis
from kafka import KafkaProducer, KafkaConsumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
RL_ACTIONS = Counter('rl_actions_total', 'Total RL actions', ['agent', 'action_type'])
RL_REWARDS = Histogram('rl_rewards', 'RL reward distribution', ['agent'])
RL_EPISODES = Counter('rl_episodes_total', 'Total RL episodes', ['agent'])
USER_ENGAGEMENT = Gauge('user_engagement_score', 'User engagement score')
CONTENT_DIVERSITY = Gauge('content_diversity_score', 'Content diversity score')
EXPLORATION_RATE = Gauge('exploration_rate', 'Current exploration rate')
REWARD_CUMULATIVE = Gauge('cumulative_reward', 'Cumulative reward')
CONVERGENCE_RATE = Gauge('convergence_rate', 'Model convergence rate')
BANDIT_REGRET = Gauge('bandit_regret', 'Cumulative bandit regret')

@dataclass
class UserProfile:
    """User profile for personalization."""
    user_id: str
    demographics: Dict[str, Any]
    interests: List[str]
    reading_history: List[str]
    engagement_patterns: Dict[str, float]
    preference_vector: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    last_updated: Optional[datetime] = None

@dataclass
class NewsItem:
    """News item representation."""
    article_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    published_at: datetime
    source: str
    embedding: Optional[np.ndarray] = None
    quality_score: Optional[float] = None
    engagement_metrics: Optional[Dict[str, float]] = None

@dataclass
class UserAction:
    """User interaction with news content."""
    user_id: str
    article_id: str
    action_type: str  # 'click', 'read', 'share', 'like', 'comment'
    timestamp: datetime
    duration: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class RecommendationState:
    """State representation for RL agent."""
    user_profile: UserProfile
    available_articles: List[NewsItem]
    time_context: Dict[str, Any]
    session_context: Dict[str, Any]
    historical_interactions: List[UserAction]

@dataclass
class RecommendationAction:
    """Action taken by RL agent."""
    recommended_articles: List[str]
    ranking_scores: List[float]
    exploration_factor: float
    diversity_factor: float
    timestamp: datetime

class NewsEnvironment:
    """Reinforcement learning environment for news recommendation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_recommendations = config.get('max_recommendations', 10)
        self.max_episode_steps = config.get('max_episode_steps', 100)
        
        # State and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config.get('state_dim', 512),),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(config.get('num_articles', 1000))
        
        # Environment state
        self.current_user = None
        self.available_articles = []
        self.step_count = 0
        self.episode_reward = 0.0
        self.interaction_history = deque(maxlen=1000)
        
        # Reward configuration
        self.reward_weights = {
            'click': 1.0,
            'read': 2.0,
            'share': 3.0,
            'like': 2.5,
            'comment': 4.0,
            'diversity_bonus': 0.5,
            'novelty_bonus': 0.3
        }
        
        logger.info("News environment initialized")
    
    def reset(self, user_profile: Optional[UserProfile] = None) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_user = user_profile or self._generate_random_user()
        self.available_articles = self._load_available_articles()
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done, info."""
        # Execute recommendation action
        recommended_article = self.available_articles[action % len(self.available_articles)]
        
        # Simulate user interaction
        user_action = self._simulate_user_interaction(recommended_article)
        
        # Calculate reward
        reward = self._calculate_reward(user_action, recommended_article)
        
        # Update state
        self.interaction_history.append(user_action)
        self.step_count += 1
        self.episode_reward += reward
        
        # Check if episode is done
        done = self.step_count >= self.max_episode_steps
        
        # Additional info
        info = {
            'user_action': user_action,
            'article': recommended_article,
            'episode_reward': self.episode_reward,
            'step_count': self.step_count
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        try:
            # User profile features
            user_features = self._encode_user_profile(self.current_user)
            
            # Article features (top articles)
            article_features = self._encode_articles(self.available_articles[:50])
            
            # Context features
            context_features = self._encode_context()
            
            # Interaction history features
            history_features = self._encode_interaction_history()
            
            # Combine all features
            state = np.concatenate([
                user_features,
                article_features.flatten(),
                context_features,
                history_features
            ])
            
            # Pad or truncate to fixed size
            target_size = self.observation_space.shape[0]
            if len(state) > target_size:
                state = state[:target_size]
            else:
                state = np.pad(state, (0, target_size - len(state)))
            
            return state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"State encoding failed: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _encode_user_profile(self, user: UserProfile) -> np.ndarray:
        """Encode user profile to feature vector."""
        features = []
        
        # Demographics (simplified)
        features.extend([1.0, 0.5, 0.3])  # age_group, gender, location
        
        # Interest categories (one-hot encoded)
        categories = ['politics', 'sports', 'technology', 'entertainment', 'business']
        for category in categories:
            features.append(1.0 if category in user.interests else 0.0)
        
        # Engagement patterns
        features.extend([
            user.engagement_patterns.get('avg_read_time', 0.0) / 300.0,  # Normalized
            user.engagement_patterns.get('click_rate', 0.0),
            user.engagement_patterns.get('share_rate', 0.0)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _encode_articles(self, articles: List[NewsItem]) -> np.ndarray:
        """Encode articles to feature matrix."""
        if not articles:
            return np.zeros((10, 20), dtype=np.float32)  # Default shape
        
        features = []
        for article in articles[:10]:  # Top 10 articles
            article_features = [
                len(article.title) / 100.0,  # Title length (normalized)
                len(article.content) / 1000.0,  # Content length (normalized)
                article.quality_score or 0.5,  # Quality score
                1.0 if article.category == 'politics' else 0.0,
                1.0 if article.category == 'sports' else 0.0,
                1.0 if article.category == 'technology' else 0.0,
                1.0 if article.category == 'entertainment' else 0.0,
                1.0 if article.category == 'business' else 0.0,
                # Engagement metrics
                article.engagement_metrics.get('click_rate', 0.0) if article.engagement_metrics else 0.0,
                article.engagement_metrics.get('share_rate', 0.0) if article.engagement_metrics else 0.0,
                # Time features
                (datetime.now() - article.published_at).total_seconds() / 86400.0,  # Days old
                # Source features
                hash(article.source) % 1000 / 1000.0,  # Source hash (normalized)
                # Tag features
                len(article.tags) / 10.0,  # Number of tags (normalized)
                # Additional features
                0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6  # Placeholder features
            ]
            features.append(article_features)
        
        # Pad if necessary
        while len(features) < 10:
            features.append([0.0] * 20)
        
        return np.array(features, dtype=np.float32)
    
    def _encode_context(self) -> np.ndarray:
        """Encode contextual information."""
        now = datetime.now()
        
        features = [
            now.hour / 24.0,  # Hour of day
            now.weekday() / 7.0,  # Day of week
            (now.day - 1) / 31.0,  # Day of month
            (now.month - 1) / 12.0,  # Month
            self.step_count / self.max_episode_steps,  # Episode progress
            len(self.interaction_history) / 1000.0,  # History length
            self.episode_reward / 100.0,  # Current episode reward
            random.random()  # Random noise
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _encode_interaction_history(self) -> np.ndarray:
        """Encode recent interaction history."""
        if not self.interaction_history:
            return np.zeros(20, dtype=np.float32)
        
        # Recent interactions (last 10)
        recent_interactions = list(self.interaction_history)[-10:]
        
        features = []
        action_types = ['click', 'read', 'share', 'like', 'comment']
        
        for action_type in action_types:
            count = sum(1 for action in recent_interactions if action.action_type == action_type)
            features.append(count / 10.0)  # Normalized count
        
        # Average duration
        durations = [action.duration for action in recent_interactions if action.duration]
        avg_duration = np.mean(durations) / 300.0 if durations else 0.0  # Normalized
        features.append(avg_duration)
        
        # Time since last interaction
        if recent_interactions:
            time_diff = (datetime.now() - recent_interactions[-1].timestamp).total_seconds() / 3600.0
            features.append(min(time_diff, 24.0) / 24.0)  # Normalized hours
        else:
            features.append(1.0)
        
        # Diversity of recent interactions
        categories = [action.context.get('category', 'unknown') for action in recent_interactions if action.context]
        unique_categories = len(set(categories)) / 5.0 if categories else 0.0
        features.append(unique_categories)
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _simulate_user_interaction(self, article: NewsItem) -> UserAction:
        """Simulate user interaction with recommended article."""
        # Simple simulation based on user preferences and article quality
        user_interest_match = self._calculate_interest_match(article)
        article_quality = article.quality_score or 0.5
        
        # Probability of different actions
        click_prob = 0.3 * user_interest_match + 0.2 * article_quality
        read_prob = 0.5 * click_prob if random.random() < click_prob else 0.0
        share_prob = 0.1 * read_prob if random.random() < read_prob else 0.0
        like_prob = 0.3 * read_prob if random.random() < read_prob else 0.0
        
        # Determine action
        if random.random() < share_prob:
            action_type = 'share'
            duration = random.uniform(60, 300)  # 1-5 minutes
        elif random.random() < like_prob:
            action_type = 'like'
            duration = random.uniform(30, 180)  # 30s-3min
        elif random.random() < read_prob:
            action_type = 'read'
            duration = random.uniform(30, 600)  # 30s-10min
        elif random.random() < click_prob:
            action_type = 'click'
            duration = random.uniform(5, 30)  # 5-30 seconds
        else:
            action_type = 'ignore'
            duration = 0.0
        
        return UserAction(
            user_id=self.current_user.user_id,
            article_id=article.article_id,
            action_type=action_type,
            timestamp=datetime.now(),
            duration=duration,
            context={'category': article.category, 'source': article.source}
        )
    
    def _calculate_interest_match(self, article: NewsItem) -> float:
        """Calculate how well article matches user interests."""
        if not self.current_user.interests:
            return 0.5
        
        # Simple keyword matching
        article_text = (article.title + ' ' + article.content).lower()
        matches = sum(1 for interest in self.current_user.interests 
                     if interest.lower() in article_text)
        
        return min(matches / len(self.current_user.interests), 1.0)
    
    def _calculate_reward(self, user_action: UserAction, article: NewsItem) -> float:
        """Calculate reward for the action."""
        base_reward = self.reward_weights.get(user_action.action_type, 0.0)
        
        # Duration bonus
        if user_action.duration:
            duration_bonus = min(user_action.duration / 300.0, 1.0)  # Max 5 minutes
            base_reward *= (1.0 + duration_bonus)
        
        # Quality bonus
        quality_bonus = (article.quality_score or 0.5) * 0.5
        
        # Diversity bonus (if article is from different category than recent)
        diversity_bonus = 0.0
        if len(self.interaction_history) > 0:
            recent_categories = [action.context.get('category') for action in list(self.interaction_history)[-5:] 
                               if action.context]
            if article.category not in recent_categories:
                diversity_bonus = self.reward_weights['diversity_bonus']
        
        # Novelty bonus (if article is recent)
        novelty_bonus = 0.0
        hours_old = (datetime.now() - article.published_at).total_seconds() / 3600.0
        if hours_old < 24:  # Less than 24 hours old
            novelty_bonus = self.reward_weights['novelty_bonus'] * (1.0 - hours_old / 24.0)
        
        total_reward = base_reward + quality_bonus + diversity_bonus + novelty_bonus
        
        return total_reward
    
    def _generate_random_user(self) -> UserProfile:
        """Generate random user profile for simulation."""
        categories = ['politics', 'sports', 'technology', 'entertainment', 'business']
        
        return UserProfile(
            user_id=str(uuid.uuid4()),
            demographics={'age_group': random.choice(['18-25', '26-35', '36-50', '50+']),
                         'gender': random.choice(['M', 'F', 'O']),
                         'location': random.choice(['US', 'EU', 'ASIA'])},
            interests=random.sample(categories, random.randint(1, 3)),
            reading_history=[],
            engagement_patterns={
                'avg_read_time': random.uniform(60, 300),
                'click_rate': random.uniform(0.1, 0.8),
                'share_rate': random.uniform(0.01, 0.2)
            }
        )
    
    def _load_available_articles(self) -> List[NewsItem]:
        """Load available articles for recommendation."""
        # Generate sample articles
        categories = ['politics', 'sports', 'technology', 'entertainment', 'business']
        sources = ['CNN', 'BBC', 'Reuters', 'TechCrunch', 'ESPN']
        
        articles = []
        for i in range(100):  # Generate 100 sample articles
            category = random.choice(categories)
            source = random.choice(sources)
            
            article = NewsItem(
                article_id=str(uuid.uuid4()),
                title=f"Sample {category} news article {i}",
                content=f"This is sample content for {category} article {i}. " * 10,
                category=category,
                tags=[category, source.lower()],
                published_at=datetime.now() - timedelta(hours=random.randint(0, 72)),
                source=source,
                quality_score=random.uniform(0.3, 1.0),
                engagement_metrics={
                    'click_rate': random.uniform(0.05, 0.5),
                    'share_rate': random.uniform(0.01, 0.1)
                }
            )
            articles.append(article)
        
        return articles

class DQNAgent(nn.Module):
    """Deep Q-Network agent for news recommendation."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Network architecture
        hidden_dims = config.get('hidden_dims', [512, 256, 128])
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.2))
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Experience replay
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        self.batch_size = config.get('batch_size', 32)
        
        # Training parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Target network for stable training
        self.target_network = deepcopy(self.network)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.update_count = 0
        
        logger.info(f"DQN Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Exploration
            action = random.randint(0, self.action_dim - 1)
            RL_ACTIONS.labels(agent='dqn', action_type='exploration').inc()
        else:
            # Exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                action = q_values.argmax().item()
            RL_ACTIONS.labels(agent='dqn', action_type='exploitation').inc()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update metrics
        EXPLORATION_RATE.set(self.epsilon)
        
        return loss.item()

class ContextualBandit:
    """Contextual bandit for personalized recommendations."""
    
    def __init__(self, num_arms: int, context_dim: int, config: Dict[str, Any]):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.config = config
        
        # Linear bandit parameters
        self.alpha = config.get('alpha', 1.0)  # Exploration parameter
        self.lambda_reg = config.get('lambda_reg', 1.0)  # Regularization
        
        # Initialize parameters for each arm
        self.A = [np.eye(context_dim) * self.lambda_reg for _ in range(num_arms)]
        self.b = [np.zeros(context_dim) for _ in range(num_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(num_arms)]
        
        # Tracking
        self.total_reward = 0.0
        self.num_pulls = [0] * num_arms
        self.regret_history = []
        
        logger.info(f"Contextual Bandit initialized with {num_arms} arms")
    
    def select_arm(self, context: np.ndarray) -> int:
        """Select arm using Upper Confidence Bound."""
        ucb_values = []
        
        for arm in range(self.num_arms):
            # Update theta (ridge regression solution)
            try:
                self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
            except np.linalg.LinAlgError:
                self.theta[arm] = np.zeros(self.context_dim)
            
            # Calculate UCB
            mean_reward = np.dot(self.theta[arm], context)
            
            # Confidence interval
            try:
                A_inv = np.linalg.inv(self.A[arm])
                confidence = self.alpha * np.sqrt(np.dot(context, np.dot(A_inv, context)))
            except np.linalg.LinAlgError:
                confidence = self.alpha
            
            ucb_value = mean_reward + confidence
            ucb_values.append(ucb_value)
        
        selected_arm = np.argmax(ucb_values)
        self.num_pulls[selected_arm] += 1
        
        RL_ACTIONS.labels(agent='bandit', action_type='selection').inc()
        
        return selected_arm
    
    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update bandit parameters with observed reward."""
        # Update sufficient statistics
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Update total reward
        self.total_reward += reward
        
        # Calculate regret (simplified)
        optimal_reward = max(np.dot(theta, context) for theta in self.theta)
        current_reward = np.dot(self.theta[arm], context)
        regret = optimal_reward - current_reward
        self.regret_history.append(regret)
        
        # Update metrics
        RL_REWARDS.labels(agent='bandit').observe(reward)
        BANDIT_REGRET.set(sum(self.regret_history))
    
    def get_arm_statistics(self) -> Dict[str, Any]:
        """Get statistics for all arms."""
        return {
            'num_pulls': self.num_pulls,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(sum(self.num_pulls), 1),
            'cumulative_regret': sum(self.regret_history),
            'theta_norms': [np.linalg.norm(theta) for theta in self.theta]
        }

class MultiAgentNewsRL:
    """Multi-agent reinforcement learning system for news recommendation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Environment
        self.env = NewsEnvironment(config)
        
        # Agents
        self.dqn_agent = DQNAgent(
            state_dim=config.get('state_dim', 512),
            action_dim=config.get('num_articles', 1000),
            config=config
        )
        
        self.bandit_agent = ContextualBandit(
            num_arms=config.get('num_articles', 1000),
            context_dim=config.get('context_dim', 128),
            config=config
        )
        
        # Agent selection strategy
        self.agent_weights = {'dqn': 0.7, 'bandit': 0.3}
        self.agent_performance = {'dqn': deque(maxlen=100), 'bandit': deque(maxlen=100)}
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.training_history = []
        
        # User modeling
        self.user_embeddings = {}
        self.user_clusters = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Multi-agent RL system initialized")
    
    def train(self, num_episodes: int = 1000):
        """Train the multi-agent system."""
        logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            while True:
                # Select agent and action
                selected_agent = self._select_agent()
                
                if selected_agent == 'dqn':
                    action = self.dqn_agent.select_action(state, training=True)
                else:  # bandit
                    context = self._extract_context(state)
                    action = self.bandit_agent.select_arm(context)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience and update agents
                if selected_agent == 'dqn':
                    self.dqn_agent.store_experience(state, action, reward, next_state, done)
                    loss = self.dqn_agent.train_step()
                    self.agent_performance['dqn'].append(reward)
                else:  # bandit
                    context = self._extract_context(state)
                    self.bandit_agent.update(action, context, reward)
                    self.agent_performance['bandit'].append(reward)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                
                if done:
                    break
            
            # Episode completed
            self.episode_count += 1
            episode_time = time.time() - episode_start_time
            
            # Update metrics
            RL_EPISODES.labels(agent='multi').inc()
            REWARD_CUMULATIVE.set(episode_reward)
            USER_ENGAGEMENT.set(self._calculate_engagement_score())
            CONTENT_DIVERSITY.set(self._calculate_diversity_score())
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean([ep['reward'] for ep in self.training_history[-100:]])
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, Steps = {episode_steps}")
                
                # Update agent weights based on performance
                self._update_agent_weights()
            
            # Store episode data
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'time': episode_time,
                'selected_agents': self._get_agent_usage_stats()
            })
        
        logger.info("Training completed")
    
    def recommend(self, user_profile: UserProfile, num_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a user."""
        try:
            # Reset environment with user
            state = self.env.reset(user_profile)
            
            recommendations = []
            
            for _ in range(num_recommendations):
                # Select best agent for this user
                selected_agent = self._select_agent_for_user(user_profile)
                
                if selected_agent == 'dqn':
                    action = self.dqn_agent.select_action(state, training=False)
                else:  # bandit
                    context = self._extract_context(state)
                    action = self.bandit_agent.select_arm(context)
                
                # Get recommended article
                if action < len(self.env.available_articles):
                    article = self.env.available_articles[action]
                    recommendations.append(article.article_id)
                    
                    # Update state (simulate recommendation)
                    next_state, _, _, _ = self.env.step(action)
                    state = next_state
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _select_agent(self) -> str:
        """Select which agent to use for current step."""
        # Weighted random selection based on recent performance
        if random.random() < self.agent_weights['dqn']:
            return 'dqn'
        else:
            return 'bandit'
    
    def _select_agent_for_user(self, user_profile: UserProfile) -> str:
        """Select best agent for specific user."""
        # For new users, prefer bandit (faster adaptation)
        if len(user_profile.reading_history) < 10:
            return 'bandit'
        
        # For experienced users, prefer DQN (better long-term modeling)
        return 'dqn'
    
    def _extract_context(self, state: np.ndarray) -> np.ndarray:
        """Extract context features for bandit agent."""
        # Use subset of state as context
        context_dim = self.config.get('context_dim', 128)
        return state[:context_dim]
    
    def _update_agent_weights(self):
        """Update agent selection weights based on performance."""
        if len(self.agent_performance['dqn']) > 10 and len(self.agent_performance['bandit']) > 10:
            dqn_avg = np.mean(list(self.agent_performance['dqn']))
            bandit_avg = np.mean(list(self.agent_performance['bandit']))
            
            total_performance = dqn_avg + bandit_avg
            if total_performance > 0:
                self.agent_weights['dqn'] = dqn_avg / total_performance
                self.agent_weights['bandit'] = bandit_avg / total_performance
    
    def _get_agent_usage_stats(self) -> Dict[str, int]:
        """Get agent usage statistics."""
        return {
            'dqn_usage': len(self.agent_performance['dqn']),
            'bandit_usage': len(self.agent_performance['bandit'])
        }
    
    def _calculate_engagement_score(self) -> float:
        """Calculate overall user engagement score."""
        if not self.env.interaction_history:
            return 0.0
        
        recent_interactions = list(self.env.interaction_history)[-50:]
        
        # Weight different action types
        action_weights = {'click': 1, 'read': 2, 'share': 4, 'like': 3, 'comment': 5}
        
        total_score = sum(action_weights.get(action.action_type, 0) 
                         for action in recent_interactions)
        
        return total_score / len(recent_interactions) if recent_interactions else 0.0
    
    def _calculate_diversity_score(self) -> float:
        """Calculate content diversity score."""
        if not self.env.interaction_history:
            return 0.0
        
        recent_interactions = list(self.env.interaction_history)[-20:]
        categories = [action.context.get('category') for action in recent_interactions 
                     if action.context and action.context.get('category')]
        
        if not categories:
            return 0.0
        
        unique_categories = len(set(categories))
        return unique_categories / len(categories)
    
    def save_model(self, filepath: str):
        """Save trained models."""
        try:
            model_data = {
                'dqn_state_dict': self.dqn_agent.state_dict(),
                'bandit_params': {
                    'A': self.bandit_agent.A,
                    'b': self.bandit_agent.b,
                    'theta': self.bandit_agent.theta
                },
                'agent_weights': self.agent_weights,
                'training_history': self.training_history,
                'config': self.config
            }
            
            torch.save(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    def load_model(self, filepath: str):
        """Load trained models."""
        try:
            model_data = torch.load(filepath)
            
            self.dqn_agent.load_state_dict(model_data['dqn_state_dict'])
            
            bandit_params = model_data['bandit_params']
            self.bandit_agent.A = bandit_params['A']
            self.bandit_agent.b = bandit_params['b']
            self.bandit_agent.theta = bandit_params['theta']
            
            self.agent_weights = model_data['agent_weights']
            self.training_history = model_data['training_history']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")

def create_rl_api(rl_system: MultiAgentNewsRL) -> Flask:
    """Create Flask API for RL system."""
    app = Flask(__name__)
    
    @app.route('/rl/recommend', methods=['POST'])
    def get_recommendations():
        try:
            data = request.get_json()
            
            # Create user profile from request
            user_profile = UserProfile(
                user_id=data.get('user_id', str(uuid.uuid4())),
                demographics=data.get('demographics', {}),
                interests=data.get('interests', []),
                reading_history=data.get('reading_history', []),
                engagement_patterns=data.get('engagement_patterns', {})
            )
            
            # Generate recommendations
            recommendations = rl_system.recommend(
                user_profile,
                num_recommendations=data.get('num_recommendations', 10)
            )
            
            return jsonify({
                'user_id': user_profile.user_id,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat(),
                'agent_weights': rl_system.agent_weights
            })
            
        except Exception as e:
            logger.error(f"Recommendation API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/rl/train', methods=['POST'])
    def start_training():
        try:
            data = request.get_json()
            num_episodes = data.get('num_episodes', 100)
            
            # Start training in background
            def train_async():
                rl_system.train(num_episodes)
            
            thread = threading.Thread(target=train_async)
            thread.start()
            
            return jsonify({
                'status': 'training_started',
                'num_episodes': num_episodes,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Training API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/rl/stats', methods=['GET'])
    def get_statistics():
        try:
            dqn_stats = {
                'epsilon': rl_system.dqn_agent.epsilon,
                'memory_size': len(rl_system.dqn_agent.memory),
                'update_count': rl_system.dqn_agent.update_count
            }
            
            bandit_stats = rl_system.bandit_agent.get_arm_statistics()
            
            return jsonify({
                'episode_count': rl_system.episode_count,
                'total_steps': rl_system.total_steps,
                'agent_weights': rl_system.agent_weights,
                'dqn_stats': dqn_stats,
                'bandit_stats': bandit_stats,
                'engagement_score': rl_system._calculate_engagement_score(),
                'diversity_score': rl_system._calculate_diversity_score()
            })
            
        except Exception as e:
            logger.error(f"Stats API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/rl/status', methods=['GET'])
    def rl_status():
        return jsonify({
            'rl_available': True,
            'agents': ['dqn', 'bandit'],
            'training_episodes': len(rl_system.training_history),
            'timestamp': datetime.now().isoformat()
        })
    
    return app

def main():
    """Main function to run the RL system."""
    # Configuration
    config = {
        'state_dim': 512,
        'context_dim': 128,
        'num_articles': 1000,
        'max_recommendations': 10,
        'max_episode_steps': 50,
        'hidden_dims': [512, 256, 128],
        'memory_size': 10000,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'gamma': 0.99,
        'learning_rate': 0.001,
        'target_update_freq': 100,
        'alpha': 1.0,
        'lambda_reg': 1.0
    }
    
    # Start Prometheus metrics server
    start_http_server(8006)
    
    # Initialize RL system
    rl_system = MultiAgentNewsRL(config)
    
    # Create and run API
    app = create_rl_api(rl_system)
    
    logger.info("Reinforcement Learning system started")
    logger.info("API available at http://localhost:5004")
    logger.info("Metrics available at http://localhost:8006")
    
    try:
        app.run(host='0.0.0.0', port=5004, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down RL system")

if __name__ == "__main__":
    main()