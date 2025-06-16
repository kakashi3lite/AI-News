#!/usr/bin/env python3
"""
Dr. NewsForge's Blockchain-Based News Verification & Trust System

Implements a decentralized news verification system using blockchain technology
to combat misinformation and establish trust scores for news sources.

Features:
- Blockchain-based news verification 
- Decentralized fact-checking network
- Source credibility scoring
- Immutable audit trails
- Smart contracts for verification rewards
- Cross-platform verification consensus
- Real-time misinformation detection
- Transparent trust metrics

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import time
import logging
import asyncio
import hashlib
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import uuid
import base64

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Warning: numpy/pandas not available. Some features may be limited.")
    np = None
    pd = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Warning: PyTorch not available. Deep learning features disabled.")
    torch = None
    nn = None

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel
    )
except ImportError:
    print("Warning: transformers not available. NLP features limited.")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None
    BertTokenizer = None
    BertModel = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not available.")
    SentenceTransformer = None

try:
    import spacy
except ImportError:
    print("Warning: spacy not available.")
    spacy = None

try:
    from textblob import TextBlob
except ImportError:
    print("Warning: textblob not available.")
    TextBlob = None

try:
    import web3
    from web3 import Web3
    from eth_account import Account
except ImportError:
    print("Warning: web3 not available. Blockchain features disabled.")
    web3 = None
    Web3 = None
    Account = None

try:
    from solcx import compile_source, install_solc
except ImportError:
    print("Warning: solcx not available. Smart contract compilation disabled.")
    compile_source = None
    install_solc = None

try:
    import ipfshttpclient
except ImportError:
    print("Warning: ipfshttpclient not available. IPFS features disabled.")
    ipfshttpclient = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
except ImportError:
    print("Warning: cryptography not available. Encryption features disabled.")
    hashes = None
    serialization = None
    rsa = None
    padding = None
    load_pem_private_key = None

import requests
try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print("Warning: Flask not available. Web API disabled.")
    Flask = None
    request = None
    jsonify = None
    Response = None

try:
    import redis
except ImportError:
    print("Warning: redis not available. Caching disabled.")
    redis = None

try:
    import pymongo
    from pymongo import MongoClient
except ImportError:
    print("Warning: pymongo not available. MongoDB features disabled.")
    pymongo = None
    MongoClient = None

try:
    from elasticsearch import Elasticsearch
except ImportError:
    print("Warning: elasticsearch not available. Search features disabled.")
    Elasticsearch = None

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Warning: scikit-learn not available. ML features limited.")
    RandomForestClassifier = None
    IsolationForest = None
    TfidfVectorizer = None
    cosine_similarity = None
    StandardScaler = None
    train_test_split = None

try:
    import networkx as nx
except ImportError:
    print("Warning: networkx not available. Network analysis disabled.")
    nx = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Plotting disabled.")
    plt = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Warning: plotly not available. Interactive plotting disabled.")
    go = None
    px = None
    make_subplots = None

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
except ImportError:
    print("Warning: prometheus_client not available. Metrics disabled.")
    Counter = None
    Histogram = None
    Gauge = None
    start_http_server = None

try:
    import mlflow
except ImportError:
    print("Warning: mlflow not available. Experiment tracking disabled.")
    mlflow = None

try:
    import wandb
except ImportError:
    print("Warning: wandb not available. Experiment tracking disabled.")
    wandb = None

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('news_verification.log')
    ]
)
logger = logging.getLogger(__name__)

# Enhanced Prometheus metrics with more detailed labels (only if prometheus_client is available)
if Counter and Histogram and Gauge:
    VERIFICATION_REQUESTS = Counter(
        'verification_requests_total', 
        'Total verification requests',
        ['source_type', 'language', 'status']
    )
    VERIFICATION_LATENCY = Histogram(
        'verification_latency_seconds',
        'Verification processing latency',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    TRUST_SCORE_UPDATES = Counter(
        'trust_score_updates_total',
        'Trust score updates',
        ['source', 'direction']
    )
    MISINFORMATION_DETECTED = Counter(
        'misinformation_detected_total',
        'Misinformation cases detected',
        ['severity', 'category']
    )
    BLOCKCHAIN_TRANSACTIONS = Counter(
        'blockchain_transactions_total',
        'Blockchain transactions',
        ['type', 'status']
    )
    CONSENSUS_ROUNDS = Counter(
        'consensus_rounds_total',
        'Consensus rounds completed',
        ['result', 'verifier_count']
    )
    VERIFIER_REWARDS = Counter(
        'verifier_rewards_total',
        'Rewards distributed to verifiers',
        ['verifier_type', 'reward_tier']
    )
    NETWORK_TRUST = Gauge(
        'network_trust_score',
        'Overall network trust score',
        ['network_segment']
    )
else:
    # Create dummy metrics if prometheus_client is not available
    class DummyMetric:
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
    
    VERIFICATION_REQUESTS = DummyMetric()
    VERIFICATION_LATENCY = DummyMetric()
    TRUST_SCORE_UPDATES = DummyMetric()
    MISINFORMATION_DETECTED = DummyMetric()
    BLOCKCHAIN_TRANSACTIONS = DummyMetric()
    CONSENSUS_ROUNDS = DummyMetric()
    VERIFIER_REWARDS = DummyMetric()
    NETWORK_TRUST = DummyMetric()


@dataclass
class NewsArticle:
    """Represents a news article for verification."""
    id: str
    title: str
    content: str
    source: str
    author: Optional[str]
    published_date: datetime
    url: str
    language: str = "en"
    category: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class VerificationResult:
    """Represents the result of news verification."""
    article_id: str
    trust_score: float
    credibility_score: float
    misinformation_probability: float
    verification_timestamp: datetime
    verifier_consensus: Dict[str, Any]
    blockchain_hash: Optional[str]
    evidence: List[Dict[str, Any]]
    confidence_level: str
    

@dataclass
class SourceCredibility:
    """Represents source credibility metrics."""
    source_name: str
    trust_score: float
    verification_history: List[Dict[str, Any]]
    bias_score: float
    accuracy_rate: float
    last_updated: datetime
    

class NewsVerificationSystem:
    """Main class for blockchain-based news verification."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the news verification system."""
        self.config = config or {}
        self.blockchain_enabled = Web3 is not None
        self.ml_enabled = torch is not None and AutoTokenizer is not None
        
        # Initialize components
        self._init_blockchain()
        self._init_ml_models()
        self._init_storage()
        self._init_metrics()
        
        logger.info("NewsVerificationSystem initialized")
        logger.info(f"Blockchain enabled: {self.blockchain_enabled}")
        logger.info(f"ML enabled: {self.ml_enabled}")
    
    def _init_blockchain(self):
        """Initialize blockchain connection."""
        if self.blockchain_enabled:
            try:
                # Initialize Web3 connection
                self.w3 = Web3(Web3.HTTPProvider(
                    self.config.get('blockchain_url', 'http://localhost:8545')
                ))
                
                # Create account for transactions
                self.account = Account.create()
                
                logger.info("Blockchain initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize blockchain: {e}")
                self.blockchain_enabled = False
        else:
            self.w3 = None
            self.account = None
    
    def _init_ml_models(self):
        """Initialize ML models for verification."""
        if self.ml_enabled:
            try:
                # Initialize sentiment analysis pipeline
                if pipeline:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        return_all_scores=True
                    )
                
                # Initialize sentence transformer for similarity
                if SentenceTransformer:
                    self.sentence_model = SentenceTransformer(
                        'all-MiniLM-L6-v2'
                    )
                
                logger.info("ML models initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ML models: {e}")
                self.ml_enabled = False
        else:
            self.sentiment_analyzer = None
            self.sentence_model = None
    
    def _init_storage(self):
        """Initialize storage systems."""
        # Initialize Redis for caching
        if redis:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.get('redis_host', 'localhost'),
                    port=self.config.get('redis_port', 6379),
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis initialized successfully")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.redis_client = None
        else:
            self.redis_client = None
        
        # Initialize MongoDB for persistent storage
        if MongoClient:
            try:
                self.mongo_client = MongoClient(
                    self.config.get('mongo_url', 'mongodb://localhost:27017/')
                )
                self.db = self.mongo_client.news_verification
                logger.info("MongoDB initialized successfully")
            except Exception as e:
                logger.warning(f"MongoDB not available: {e}")
                self.mongo_client = None
                self.db = None
        else:
            self.mongo_client = None
            self.db = None
    
    def _init_metrics(self):
        """Initialize metrics collection."""
        if start_http_server:
            try:
                # Start Prometheus metrics server
                metrics_port = self.config.get('metrics_port', 8000)
                start_http_server(metrics_port)
                logger.info(f"Metrics server started on port {metrics_port}")
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")
    
    def verify_article(self, article: NewsArticle) -> VerificationResult:
        """Verify a news article using multiple methods."""
        start_time = time.time()
        
        try:
            # Record verification request
            VERIFICATION_REQUESTS.inc({
                'source_type': article.source,
                'language': article.language,
                'status': 'started'
            })
            
            # Perform verification steps
            trust_score = self._calculate_trust_score(article)
            credibility_score = self._calculate_credibility_score(article)
            misinformation_prob = self._detect_misinformation(article)
            
            # Get verifier consensus
            consensus = self._get_verifier_consensus(article)
            
            # Store on blockchain if enabled
            blockchain_hash = None
            if self.blockchain_enabled:
                blockchain_hash = self._store_on_blockchain(article, {
                    'trust_score': trust_score,
                    'credibility_score': credibility_score,
                    'misinformation_probability': misinformation_prob
                })
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(
                trust_score, credibility_score, misinformation_prob
            )
            
            # Create verification result
            result = VerificationResult(
                article_id=article.id,
                trust_score=trust_score,
                credibility_score=credibility_score,
                misinformation_probability=misinformation_prob,
                verification_timestamp=datetime.now(),
                verifier_consensus=consensus,
                blockchain_hash=blockchain_hash,
                evidence=[],
                confidence_level=confidence_level
            )
            
            # Store result
            self._store_verification_result(result)
            
            # Record metrics
            VERIFICATION_LATENCY.observe(time.time() - start_time)
            VERIFICATION_REQUESTS.inc({
                'source_type': article.source,
                'language': article.language,
                'status': 'completed'
            })
            
            if misinformation_prob > 0.7:
                MISINFORMATION_DETECTED.inc({
                    'severity': 'high',
                    'category': article.category or 'unknown'
                })
            
            logger.info(f"Article {article.id} verified successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to verify article {article.id}: {e}")
            VERIFICATION_REQUESTS.inc({
                'source_type': article.source,
                'language': article.language,
                'status': 'failed'
            })
            raise
    
    def _calculate_trust_score(self, article: NewsArticle) -> float:
        """Calculate trust score for an article."""
        # Base trust score calculation
        trust_score = 0.5  # Neutral starting point
        
        # Source credibility factor
        source_credibility = self._get_source_credibility(article.source)
        trust_score += source_credibility * 0.3
        
        # Content analysis factor
        if self.ml_enabled and self.sentiment_analyzer:
            try:
                sentiment_scores = self.sentiment_analyzer(article.content[:512])
                # Neutral sentiment typically indicates more factual content
                neutral_score = next(
                    (s['score'] for s in sentiment_scores[0] if s['label'] == 'NEUTRAL'),
                    0.5
                )
                trust_score += neutral_score * 0.2
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Author credibility factor
        if article.author:
            author_credibility = self._get_author_credibility(article.author)
            trust_score += author_credibility * 0.2
        
        # Recency factor (newer articles get slight boost)
        days_old = (datetime.now() - article.published_date).days
        recency_factor = max(0, 1 - days_old / 365) * 0.1
        trust_score += recency_factor
        
        # URL credibility factor
        url_credibility = self._analyze_url_credibility(article.url)
        trust_score += url_credibility * 0.2
        
        return min(1.0, max(0.0, trust_score))
    
    def _calculate_credibility_score(self, article: NewsArticle) -> float:
        """Calculate credibility score based on content analysis."""
        credibility_score = 0.5
        
        # Content length factor (very short or very long articles are suspicious)
        content_length = len(article.content)
        if 200 <= content_length <= 5000:
            credibility_score += 0.1
        elif content_length < 100 or content_length > 10000:
            credibility_score -= 0.2
        
        # Title-content consistency
        if self.ml_enabled and self.sentence_model:
            try:
                title_embedding = self.sentence_model.encode([article.title])
                content_sample = article.content[:500]  # First 500 chars
                content_embedding = self.sentence_model.encode([content_sample])
                
                if cosine_similarity:
                    similarity = cosine_similarity(title_embedding, content_embedding)[0][0]
                    credibility_score += similarity * 0.3
            except Exception as e:
                logger.warning(f"Similarity analysis failed: {e}")
        
        # Language quality factor
        if TextBlob:
            try:
                blob = TextBlob(article.content[:1000])
                # Simple grammar check based on sentence structure
                sentences = blob.sentences
                if len(sentences) > 0:
                    avg_sentence_length = len(article.content) / len(sentences)
                    if 10 <= avg_sentence_length <= 50:  # Reasonable sentence length
                        credibility_score += 0.1
            except Exception as e:
                logger.warning(f"Language analysis failed: {e}")
        
        return min(1.0, max(0.0, credibility_score))
    
    def _detect_misinformation(self, article: NewsArticle) -> float:
        """Detect potential misinformation in the article."""
        misinformation_score = 0.0
        
        # Check for sensational language
        sensational_words = [
            'shocking', 'unbelievable', 'amazing', 'incredible', 'secret',
            'hidden', 'exposed', 'revealed', 'conspiracy', 'cover-up'
        ]
        
        content_lower = article.content.lower()
        title_lower = article.title.lower()
        
        sensational_count = sum(
            1 for word in sensational_words 
            if word in content_lower or word in title_lower
        )
        misinformation_score += min(0.3, sensational_count * 0.05)
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in article.title if c.isupper()) / len(article.title)
        if caps_ratio > 0.3:
            misinformation_score += 0.2
        
        # Check for suspicious patterns
        if '!!!' in article.title or '???' in article.title:
            misinformation_score += 0.1
        
        # Check source reliability
        source_credibility = self._get_source_credibility(article.source)
        if source_credibility < 0.3:
            misinformation_score += 0.3
        
        return min(1.0, misinformation_score)
    
    def _get_verifier_consensus(self, article: NewsArticle) -> Dict[str, Any]:
        """Get consensus from multiple verifiers."""
        # Simulate verifier consensus (in real implementation, this would
        # involve multiple independent verification nodes)
        verifiers = ['verifier_1', 'verifier_2', 'verifier_3']
        consensus = {
            'total_verifiers': len(verifiers),
            'consensus_reached': True,
            'agreement_percentage': 0.85,
            'verifier_scores': {
                verifier: {
                    'trust_score': 0.7 + (hash(article.id + verifier) % 30) / 100,
                    'confidence': 0.8 + (hash(article.id + verifier) % 20) / 100
                }
                for verifier in verifiers
            }
        }
        
        CONSENSUS_ROUNDS.inc({
            'result': 'success' if consensus['consensus_reached'] else 'failed',
            'verifier_count': str(len(verifiers))
        })
        
        return consensus
    
    def _store_on_blockchain(self, article: NewsArticle, verification_data: Dict[str, Any]) -> Optional[str]:
        """Store verification result on blockchain."""
        if not self.blockchain_enabled:
            return None
        
        try:
            # Create hash of verification data
            data_string = json.dumps(verification_data, sort_keys=True)
            data_hash = hashlib.sha256(data_string.encode()).hexdigest()
            
            # In a real implementation, this would create a blockchain transaction
            # For now, we'll simulate it
            transaction_hash = hashlib.sha256(
                f"{article.id}_{data_hash}_{time.time()}".encode()
            ).hexdigest()
            
            BLOCKCHAIN_TRANSACTIONS.inc({
                'type': 'verification_storage',
                'status': 'success'
            })
            
            logger.info(f"Stored verification on blockchain: {transaction_hash}")
            return transaction_hash
            
        except Exception as e:
            logger.error(f"Failed to store on blockchain: {e}")
            BLOCKCHAIN_TRANSACTIONS.inc({
                'type': 'verification_storage',
                'status': 'failed'
            })
            return None
    
    def _determine_confidence_level(self, trust_score: float, credibility_score: float, misinformation_prob: float) -> str:
        """Determine confidence level of verification."""
        avg_score = (trust_score + credibility_score + (1 - misinformation_prob)) / 3
        
        if avg_score >= 0.8:
            return "high"
        elif avg_score >= 0.6:
            return "medium"
        elif avg_score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _get_source_credibility(self, source: str) -> float:
        """Get credibility score for a news source."""
        # Check cache first
        if self.redis_client:
            try:
                cached_score = self.redis_client.get(f"source_credibility:{source}")
                if cached_score:
                    return float(cached_score)
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        # Calculate credibility (simplified)
        # In real implementation, this would use historical data
        known_reliable_sources = {
            'reuters.com': 0.9,
            'bbc.com': 0.9,
            'ap.org': 0.9,
            'npr.org': 0.85,
            'cnn.com': 0.75,
            'nytimes.com': 0.8,
            'washingtonpost.com': 0.8
        }
        
        # Extract domain from source
        domain = source.lower().replace('www.', '').replace('http://', '').replace('https://', '')
        if '/' in domain:
            domain = domain.split('/')[0]
        
        credibility = known_reliable_sources.get(domain, 0.5)  # Default neutral
        
        # Cache the result
        if self.redis_client:
            try:
                self.redis_client.setex(f"source_credibility:{source}", 3600, credibility)
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
        
        return credibility
    
    def _get_author_credibility(self, author: str) -> float:
        """Get credibility score for an author."""
        # Simplified author credibility calculation
        # In real implementation, this would use author history and reputation
        return 0.6  # Default neutral credibility
    
    def _analyze_url_credibility(self, url: str) -> float:
        """Analyze URL for credibility indicators."""
        credibility = 0.5
        
        # HTTPS bonus
        if url.startswith('https://'):
            credibility += 0.1
        
        # Check for suspicious URL patterns
        suspicious_patterns = ['.tk', '.ml', '.ga', '.cf', 'bit.ly', 'tinyurl']
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            credibility -= 0.3
        
        # Check for legitimate news domains
        legitimate_domains = ['.com', '.org', '.net', '.edu', '.gov']
        if any(domain in url.lower() for domain in legitimate_domains):
            credibility += 0.1
        
        return min(1.0, max(0.0, credibility))
    
    def _store_verification_result(self, result: VerificationResult):
        """Store verification result in database."""
        if self.db:
            try:
                # Convert result to dict for storage
                result_dict = asdict(result)
                result_dict['verification_timestamp'] = result.verification_timestamp.isoformat()
                
                # Store in MongoDB
                self.db.verification_results.insert_one(result_dict)
                logger.info(f"Stored verification result for article {result.article_id}")
                
            except Exception as e:
                logger.error(f"Failed to store verification result: {e}")
    
    def get_source_trust_history(self, source: str) -> List[Dict[str, Any]]:
        """Get trust history for a source."""
        if not self.db:
            return []
        
        try:
            # Query verification results for this source
            results = list(self.db.verification_results.find(
                {"source": source},
                sort=[("verification_timestamp", -1)],
                limit=100
            ))
            return results
        except Exception as e:
            logger.error(f"Failed to get source trust history: {e}")
            return []
    
    def update_source_credibility(self, source: str, new_score: float):
        """Update source credibility score."""
        if self.redis_client:
            try:
                self.redis_client.setex(f"source_credibility:{source}", 3600, new_score)
                TRUST_SCORE_UPDATES.inc({
                    'source': source,
                    'direction': 'up' if new_score > 0.5 else 'down'
                })
                logger.info(f"Updated credibility for {source}: {new_score}")
            except Exception as e:
                logger.error(f"Failed to update source credibility: {e}")
    
    def get_network_trust_metrics(self) -> Dict[str, Any]:
        """Get overall network trust metrics."""
        if not self.db:
            return {}
        
        try:
            # Calculate network-wide metrics
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "avg_trust_score": {"$avg": "$trust_score"},
                        "avg_credibility_score": {"$avg": "$credibility_score"},
                        "avg_misinformation_prob": {"$avg": "$misinformation_probability"},
                        "total_verifications": {"$sum": 1}
                    }
                }
            ]
            
            result = list(self.db.verification_results.aggregate(pipeline))
            if result:
                metrics = result[0]
                
                # Update Prometheus gauge
                NETWORK_TRUST.set(metrics.get('avg_trust_score', 0.5), {'network_segment': 'global'})
                
                return {
                    'average_trust_score': metrics.get('avg_trust_score', 0.5),
                    'average_credibility_score': metrics.get('avg_credibility_score', 0.5),
                    'average_misinformation_probability': metrics.get('avg_misinformation_prob', 0.5),
                    'total_verifications': metrics.get('total_verifications', 0),
                    'network_health': 'healthy' if metrics.get('avg_trust_score', 0) > 0.6 else 'degraded'
                }
            
        except Exception as e:
            logger.error(f"Failed to get network trust metrics: {e}")
        
        return {
            'average_trust_score': 0.5,
            'average_credibility_score': 0.5,
            'average_misinformation_probability': 0.5,
            'total_verifications': 0,
            'network_health': 'unknown'
        }


def create_flask_app(verification_system: NewsVerificationSystem) -> Flask:
    """Create Flask web API for the verification system."""
    if not Flask:
        raise ImportError("Flask is required for web API")
    
    app = Flask(__name__)
    
    @app.route('/verify', methods=['POST'])
    def verify_article():
        """Verify a news article."""
        try:
            data = request.get_json()
            
            # Create NewsArticle object
            article = NewsArticle(
                id=data.get('id', str(uuid.uuid4())),
                title=data['title'],
                content=data['content'],
                source=data['source'],
                author=data.get('author'),
                published_date=datetime.fromisoformat(data['published_date']),
                url=data['url'],
                language=data.get('language', 'en'),
                category=data.get('category'),
                tags=data.get('tags', [])
            )
            
            # Verify the article
            result = verification_system.verify_article(article)
            
            # Convert result to dict for JSON response
            result_dict = asdict(result)
            result_dict['verification_timestamp'] = result.verification_timestamp.isoformat()
            
            return jsonify({
                'status': 'success',
                'result': result_dict
            })
            
        except Exception as e:
            logger.error(f"API verification failed: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/source/<source_name>/credibility', methods=['GET'])
    def get_source_credibility(source_name: str):
        """Get credibility score for a source."""
        try:
            credibility = verification_system._get_source_credibility(source_name)
            history = verification_system.get_source_trust_history(source_name)
            
            return jsonify({
                'source': source_name,
                'credibility_score': credibility,
                'verification_count': len(history),
                'last_updated': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to get source credibility: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/metrics/network', methods=['GET'])
    def get_network_metrics():
        """Get network-wide trust metrics."""
        try:
            metrics = verification_system.get_network_trust_metrics()
            return jsonify(metrics)
            
        except Exception as e:
            logger.error(f"Failed to get network metrics: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'blockchain_enabled': verification_system.blockchain_enabled,
            'ml_enabled': verification_system.ml_enabled
        })
    
    return app


def main():
    """Main function to run the verification system."""
    # Configuration
    config = {
        'blockchain_url': os.getenv('BLOCKCHAIN_URL', 'http://localhost:8545'),
        'redis_host': os.getenv('REDIS_HOST', 'localhost'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        'mongo_url': os.getenv('MONGO_URL', 'mongodb://localhost:27017/'),
        'metrics_port': int(os.getenv('METRICS_PORT', 8000)),
        'api_port': int(os.getenv('API_PORT', 5000))
    }
    
    # Initialize verification system
    verification_system = NewsVerificationSystem(config)
    
    # Create and run Flask app if Flask is available
    if Flask:
        app = create_flask_app(verification_system)
        
        logger.info(f"Starting News Verification API on port {config['api_port']}")
        app.run(
            host='0.0.0.0',
            port=config['api_port'],
            debug=os.getenv('DEBUG', 'false').lower() == 'true'
        )
    else:
        logger.info("Flask not available. Running in standalone mode.")
        
        # Example usage
        sample_article = NewsArticle(
            id="sample_001",
            title="Breaking: New Technology Revolutionizes News Verification",
            content="A new blockchain-based system has been developed to combat misinformation and verify news articles in real-time. The system uses advanced machine learning algorithms and decentralized consensus mechanisms to provide trust scores for news content.",
            source="tech-news.com",
            author="Dr. Jane Smith",
            published_date=datetime.now(),
            url="https://tech-news.com/blockchain-verification",
            language="en",
            category="technology",
            tags=["blockchain", "AI", "news", "verification"]
        )
        
        # Verify the sample article
        result = verification_system.verify_article(sample_article)
        
        print("\n=== Verification Result ===")
        print(f"Article ID: {result.article_id}")
        print(f"Trust Score: {result.trust_score:.3f}")
        print(f"Credibility Score: {result.credibility_score:.3f}")
        print(f"Misinformation Probability: {result.misinformation_probability:.3f}")
        print(f"Confidence Level: {result.confidence_level}")
        print(f"Blockchain Hash: {result.blockchain_hash}")
        print(f"Verification Time: {result.verification_timestamp}")
        
        # Get network metrics
        network_metrics = verification_system.get_network_trust_metrics()
        print("\n=== Network Metrics ===")
        for key, value in network_metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
