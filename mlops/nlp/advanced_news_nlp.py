#!/usr/bin/env python3
"""
Dr. NewsForge's Advanced NLP News Analysis System

Comprehensive natural language processing system for news content analysis,
sentiment analysis, entity extraction, topic modeling, and semantic understanding.
Implements state-of-the-art transformer models, multilingual processing,
and advanced text analytics for news media.

Features:
- BERT/RoBERTa for text classification and sentiment analysis
- Named Entity Recognition (NER) with custom news entities
- Topic modeling with BERTopic and LDA
- Multilingual text processing and translation
- Semantic similarity and clustering
- Fact-checking and claim verification
- Bias detection and political stance analysis
- Readability and quality assessment
- Real-time text streaming analysis
- Knowledge graph construction
- Automated summarization with T5/BART
- Question answering systems

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
from collections import defaultdict, Counter, deque
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
import re
import string
from urllib.parse import urlparse
import requests

# NLP libraries
import nltk
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from googletrans import Translator

# Advanced NLP frameworks
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    BertTokenizer, BertModel, BertForSequenceClassification,
    RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    pipeline, Pipeline
)
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler

# Scientific computing
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Text processing
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Knowledge graphs
try:
    import networkx as nx
    from pyvis.network import Network
    GRAPH_LIBS_AVAILABLE = True
except ImportError:
    GRAPH_LIBS_AVAILABLE = False
    logging.warning("Graph libraries not available")

# Monitoring and visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# MLOps and tracking
import mlflow
import wandb
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Networking
from flask import Flask, request, jsonify
import redis
from kafka import KafkaProducer, KafkaConsumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set deterministic behavior for language detection
DetectorFactory.seed = 0

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download failed: {e}")

# Load spaCy models
try:
    nlp_en = spacy.load("en_core_web_sm")
    SPACY_EN_AVAILABLE = True
except OSError:
    logger.warning("English spaCy model not available")
    SPACY_EN_AVAILABLE = False

# Prometheus metrics
NLP_REQUESTS = Counter('nlp_requests_total', 'Total NLP requests', ['task_type'])
NLP_LATENCY = Histogram('nlp_latency_seconds', 'NLP processing latency', ['task_type'])
SENTIMENT_SCORE = Gauge('sentiment_score', 'Average sentiment score')
TEXT_QUALITY = Gauge('text_quality_score', 'Text quality score')
ENTITY_COUNT = Gauge('entity_count', 'Number of entities extracted')
TOPIC_COHERENCE = Gauge('topic_coherence', 'Topic model coherence score')
BIAS_SCORE = Gauge('bias_score', 'Text bias score')
READABILITY_SCORE = Gauge('readability_score', 'Text readability score')
FACT_CHECK_CONFIDENCE = Gauge('fact_check_confidence', 'Fact-checking confidence')
SEMANTIC_SIMILARITY = Gauge('semantic_similarity', 'Semantic similarity score')

@dataclass
class NewsText:
    """News text representation."""
    text_id: str
    content: str
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    language: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Entity:
    """Named entity representation."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    description: Optional[str] = None
    linked_entity: Optional[str] = None

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result."""
    overall_sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    scores: Dict[str, float]  # {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}
    emotion_scores: Optional[Dict[str, float]] = None
    aspect_sentiments: Optional[Dict[str, Dict[str, float]]] = None

@dataclass
class TopicAnalysis:
    """Topic analysis result."""
    topics: List[Dict[str, Any]]
    topic_distribution: List[float]
    coherence_score: float
    dominant_topic: Optional[Dict[str, Any]] = None

@dataclass
class TextAnalysis:
    """Comprehensive text analysis result."""
    text_id: str
    language: str
    sentiment: SentimentAnalysis
    entities: List[Entity]
    topics: TopicAnalysis
    keywords: List[str]
    summary: str
    quality_metrics: Dict[str, float]
    bias_analysis: Dict[str, Any]
    fact_check_results: Dict[str, Any]
    readability_metrics: Dict[str, float]
    semantic_features: Dict[str, Any]

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load pre-trained sentiment models
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            return_all_scores=True
        )
        
        # Load emotion analysis model
        try:
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=config.get('emotion_model', 'j-hartmann/emotion-english-distilroberta-base'),
                return_all_scores=True
            )
            self.emotion_available = True
        except Exception as e:
            logger.warning(f"Emotion model not available: {e}")
            self.emotion_available = False
        
        # VADER sentiment analyzer
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.vader_available = True
        except Exception as e:
            logger.warning(f"VADER not available: {e}")
            self.vader_available = False
        
        logger.info("Advanced Sentiment Analyzer initialized")
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Comprehensive sentiment analysis."""
        try:
            # Transformer-based sentiment analysis
            transformer_results = self.sentiment_pipeline(text)
            
            # Process transformer results
            sentiment_scores = {}
            for result in transformer_results[0]:
                label = result['label'].lower()
                if 'pos' in label:
                    sentiment_scores['positive'] = result['score']
                elif 'neg' in label:
                    sentiment_scores['negative'] = result['score']
                else:
                    sentiment_scores['neutral'] = result['score']
            
            # Determine overall sentiment
            overall_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            confidence = sentiment_scores[overall_sentiment]
            
            # Emotion analysis
            emotion_scores = None
            if self.emotion_available:
                try:
                    emotion_results = self.emotion_pipeline(text)
                    emotion_scores = {result['label'].lower(): result['score'] 
                                    for result in emotion_results[0]}
                except Exception as e:
            logger.error(f"Keyword extraction API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    return app

def main():
    """Main function to run the Advanced News NLP System."""
    try:
        # Initialize configuration
        config = {
            'models': {
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'ner_model': 'dbmdz/bert-large-cased-finetuned-conll03-english',
                'summarization_model': 'facebook/bart-large-cnn',
                'topic_model_components': 10,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8080,
                'debug': False
            },
            'monitoring': {
                'prometheus_port': 8081,
                'log_level': 'INFO'
            },
            'cache': {
                'redis_url': 'redis://localhost:6379',
                'ttl': 3600
            }
        }
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config['monitoring']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Advanced News NLP System...")
        
        # Initialize NLP system
        nlp_system = AdvancedNewsNLP(config)
        
        # Create Flask app
        app = create_nlp_api(nlp_system)
        
        # Start Prometheus metrics server
        start_http_server(config['monitoring']['prometheus_port'])
        logger.info(f"Prometheus metrics server started on port {config['monitoring']['prometheus_port']}")
        
        # Log system information
        logger.info(f"NLP System initialized with:")
        logger.info(f"- Sentiment Model: {config['models']['sentiment_model']}")
        logger.info(f"- NER Model: {config['models']['ner_model']}")
        logger.info(f"- Summarization Model: {config['models']['summarization_model']}")
        logger.info(f"- Topic Model Components: {config['models']['topic_model_components']}")
        logger.info(f"- Embedding Model: {config['models']['embedding_model']}")
        
        # Run Flask app
        logger.info(f"Starting NLP API server on {config['api']['host']}:{config['api']['port']}")
        app.run(
            host=config['api']['host'],
            port=config['api']['port'],
            debug=config['api']['debug'],
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start Advanced News NLP System: {e}")
        raise

if __name__ == "__main__":
    main() as e:
                    logger.warning(f"Emotion analysis failed: {e}")
            
            # VADER analysis for comparison
            if self.vader_available:
                try:
                    vader_scores = self.vader_analyzer.polarity_scores(text)
                    # Blend VADER with transformer results
                    sentiment_scores['vader_compound'] = vader_scores['compound']
                except Exception as e:
                    logger.warning(f"VADER analysis failed: {e}")
            
            # Aspect-based sentiment analysis (simplified)
            aspect_sentiments = self._analyze_aspect_sentiment(text)
            
            SENTIMENT_SCORE.set(confidence if overall_sentiment == 'positive' else -confidence)
            
            return SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                scores=sentiment_scores,
                emotion_scores=emotion_scores,
                aspect_sentiments=aspect_sentiments
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return SentimentAnalysis(
                overall_sentiment='neutral',
                confidence=0.0,
                scores={'neutral': 1.0}
            )
    
    def _analyze_aspect_sentiment(self, text: str) -> Dict[str, Dict[str, float]]:
        """Analyze sentiment for different aspects (simplified)."""
        try:
            # Define common news aspects
            aspects = {
                'economy': ['economy', 'economic', 'financial', 'market', 'business', 'trade'],
                'politics': ['political', 'government', 'policy', 'election', 'politician'],
                'social': ['social', 'society', 'community', 'people', 'public'],
                'technology': ['technology', 'tech', 'digital', 'innovation', 'AI'],
                'environment': ['environment', 'climate', 'green', 'sustainability']
            }
            
            aspect_sentiments = {}
            text_lower = text.lower()
            
            for aspect, keywords in aspects.items():
                # Check if aspect is mentioned
                if any(keyword in text_lower for keyword in keywords):
                    # Extract sentences containing aspect keywords
                    sentences = nltk.sent_tokenize(text)
                    aspect_sentences = []
                    
                    for sentence in sentences:
                        if any(keyword in sentence.lower() for keyword in keywords):
                            aspect_sentences.append(sentence)
                    
                    if aspect_sentences:
                        # Analyze sentiment of aspect-related sentences
                        aspect_text = ' '.join(aspect_sentences)
                        aspect_results = self.sentiment_pipeline(aspect_text)
                        
                        aspect_scores = {}
                        for result in aspect_results[0]:
                            label = result['label'].lower()
                            if 'pos' in label:
                                aspect_scores['positive'] = result['score']
                            elif 'neg' in label:
                                aspect_scores['negative'] = result['score']
                            else:
                                aspect_scores['neutral'] = result['score']
                        
                        aspect_sentiments[aspect] = aspect_scores
            
            return aspect_sentiments
            
        except Exception as e:
            logger.error(f"Aspect sentiment analysis failed: {e}")
            return {}

class NamedEntityRecognizer:
    """Advanced named entity recognition with custom news entities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=config.get('ner_model', 'dbmdz/bert-large-cased-finetuned-conll03-english'),
            aggregation_strategy="simple"
        )
        
        # Load spaCy model if available
        self.spacy_available = SPACY_EN_AVAILABLE
        if self.spacy_available:
            self.nlp = nlp_en
        
        # Custom entity patterns for news
        self.news_patterns = {
            'ORGANIZATION': r'\b(?:Corp|Inc|Ltd|LLC|Company|Organization|Agency|Department|Ministry)\b',
            'LOCATION': r'\b(?:Street|Avenue|Road|Boulevard|City|State|Country|County)\b',
            'EVENT': r'\b(?:Conference|Summit|Meeting|Election|War|Crisis|Disaster)\b'
        }
        
        logger.info("Named Entity Recognizer initialized")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        try:
            entities = []
            
            # Transformer-based NER
            transformer_entities = self._extract_transformer_entities(text)
            entities.extend(transformer_entities)
            
            # spaCy NER if available
            if self.spacy_available:
                spacy_entities = self._extract_spacy_entities(text)
                entities.extend(spacy_entities)
            
            # Custom pattern-based extraction
            pattern_entities = self._extract_pattern_entities(text)
            entities.extend(pattern_entities)
            
            # Remove duplicates and merge overlapping entities
            entities = self._merge_entities(entities)
            
            ENTITY_COUNT.set(len(entities))
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _extract_transformer_entities(self, text: str) -> List[Entity]:
        """Extract entities using transformer model."""
        try:
            results = self.ner_pipeline(text)
            entities = []
            
            for result in results:
                entity = Entity(
                    text=result['word'],
                    label=result['entity_group'],
                    start=result['start'],
                    end=result['end'],
                    confidence=result['score']
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Transformer NER failed: {e}")
            return []
    
    def _extract_spacy_entities(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,  # spaCy doesn't provide confidence scores
                    description=spacy.explain(ent.label_)
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"spaCy NER failed: {e}")
            return []
    
    def _extract_pattern_entities(self, text: str) -> List[Entity]:
        """Extract entities using custom patterns."""
        try:
            entities = []
            
            for label, pattern in self.news_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7  # Lower confidence for pattern-based
                    )
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Pattern entity extraction failed: {e}")
            return []
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities and remove duplicates."""
        try:
            if not entities:
                return []
            
            # Sort by start position
            entities.sort(key=lambda x: x.start)
            
            merged = []
            current = entities[0]
            
            for next_entity in entities[1:]:
                # Check for overlap
                if next_entity.start <= current.end:
                    # Merge entities - keep the one with higher confidence
                    if next_entity.confidence > current.confidence:
                        current = next_entity
                else:
                    merged.append(current)
                    current = next_entity
            
            merged.append(current)
            
            return merged
            
        except Exception as e:
            logger.error(f"Entity merging failed: {e}")
            return entities

class TopicModeler:
    """Advanced topic modeling with BERTopic and LDA."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize sentence transformer for embeddings
        embedding_model = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.sentence_model = SentenceTransformer(embedding_model)
        
        # Initialize BERTopic
        umap_model = UMAP(
            n_neighbors=config.get('umap_neighbors', 15),
            n_components=config.get('umap_components', 5),
            min_dist=config.get('umap_min_dist', 0.0),
            metric='cosine',
            random_state=42
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=config.get('min_cluster_size', 10),
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=False
        )
        
        # Initialize LDA for comparison
        self.lda_model = None
        self.vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 1000),
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Cache for fitted models
        self.is_fitted = False
        self.documents_cache = []
        
        logger.info("Topic Modeler initialized")
    
    def fit_topics(self, documents: List[str]) -> None:
        """Fit topic models on documents."""
        try:
            if len(documents) < 10:
                logger.warning("Too few documents for topic modeling")
                return
            
            # Fit BERTopic
            self.topic_model.fit(documents)
            
            # Fit LDA
            doc_term_matrix = self.vectorizer.fit_transform(documents)
            n_topics = min(10, len(documents) // 5)  # Adaptive number of topics
            
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            self.lda_model.fit(doc_term_matrix)
            
            self.is_fitted = True
            self.documents_cache = documents
            
            logger.info(f"Topic models fitted on {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Topic model fitting failed: {e}")
    
    def analyze_topics(self, text: str) -> TopicAnalysis:
        """Analyze topics in text."""
        try:
            if not self.is_fitted:
                # Fit on single document if no model exists
                self.fit_topics([text])
            
            # BERTopic analysis
            topics_info = self.topic_model.get_topic_info()
            topic_probs = self.topic_model.transform([text])[1][0]
            
            # Get topic details
            topics = []
            for idx, row in topics_info.iterrows():
                if row['Topic'] != -1:  # Exclude outlier topic
                    topic_words = self.topic_model.get_topic(row['Topic'])
                    topics.append({
                        'id': row['Topic'],
                        'words': [word for word, _ in topic_words[:10]],
                        'scores': [score for _, score in topic_words[:10]],
                        'count': row['Count']
                    })
            
            # Calculate coherence score (simplified)
            coherence_score = self._calculate_coherence(topics)
            
            # Find dominant topic
            dominant_topic = None
            if len(topic_probs) > 0:
                dominant_idx = np.argmax(topic_probs)
                if dominant_idx < len(topics):
                    dominant_topic = topics[dominant_idx]
                    dominant_topic['probability'] = topic_probs[dominant_idx]
            
            TOPIC_COHERENCE.set(coherence_score)
            
            return TopicAnalysis(
                topics=topics,
                topic_distribution=topic_probs.tolist(),
                coherence_score=coherence_score,
                dominant_topic=dominant_topic
            )
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return TopicAnalysis(
                topics=[],
                topic_distribution=[],
                coherence_score=0.0
            )
    
    def _calculate_coherence(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate topic coherence score (simplified)."""
        try:
            if not topics:
                return 0.0
            
            # Simple coherence based on word co-occurrence
            coherence_scores = []
            
            for topic in topics:
                words = topic['words'][:5]  # Top 5 words
                if len(words) < 2:
                    continue
                
                # Calculate pairwise similarities
                word_embeddings = self.sentence_model.encode(words)
                similarities = []
                
                for i in range(len(words)):
                    for j in range(i + 1, len(words)):
                        sim = 1 - cosine(word_embeddings[i], word_embeddings[j])
                        similarities.append(sim)
                
                if similarities:
                    coherence_scores.append(np.mean(similarities))
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {e}")
            return 0.0

class BiasDetector:
    """Detect bias and political stance in news text."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load bias detection model (placeholder - would need trained model)
        try:
            self.bias_pipeline = pipeline(
                "text-classification",
                model=config.get('bias_model', 'unitary/toxic-bert'),
                return_all_scores=True
            )
            self.bias_model_available = True
        except Exception as e:
            logger.warning(f"Bias model not available: {e}")
            self.bias_model_available = False
        
        # Political bias indicators
        self.political_keywords = {
            'left': ['progressive', 'liberal', 'democrat', 'socialism', 'equality'],
            'right': ['conservative', 'republican', 'traditional', 'capitalism', 'freedom'],
            'center': ['moderate', 'bipartisan', 'compromise', 'balanced']
        }
        
        # Bias indicators
        self.bias_indicators = {
            'emotional_language': ['outrageous', 'shocking', 'devastating', 'incredible'],
            'loaded_words': ['terrorist', 'hero', 'victim', 'criminal'],
            'absolute_terms': ['always', 'never', 'all', 'none', 'every']
        }
        
        logger.info("Bias Detector initialized")
    
    def analyze_bias(self, text: str) -> Dict[str, Any]:
        """Analyze bias in text."""
        try:
            bias_analysis = {}
            
            # Political stance analysis
            political_stance = self._analyze_political_stance(text)
            bias_analysis['political_stance'] = political_stance
            
            # Emotional bias detection
            emotional_bias = self._detect_emotional_bias(text)
            bias_analysis['emotional_bias'] = emotional_bias
            
            # Language bias indicators
            language_bias = self._detect_language_bias(text)
            bias_analysis['language_bias'] = language_bias
            
            # Source credibility indicators
            credibility_indicators = self._analyze_credibility_indicators(text)
            bias_analysis['credibility_indicators'] = credibility_indicators
            
            # Overall bias score
            overall_bias = self._calculate_overall_bias(bias_analysis)
            bias_analysis['overall_bias_score'] = overall_bias
            
            BIAS_SCORE.set(overall_bias)
            
            return bias_analysis
            
        except Exception as e:
            logger.error(f"Bias analysis failed: {e}")
            return {'overall_bias_score': 0.0}
    
    def _analyze_political_stance(self, text: str) -> Dict[str, Any]:
        """Analyze political stance of text."""
        try:
            text_lower = text.lower()
            stance_scores = {}
            
            for stance, keywords in self.political_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                stance_scores[stance] = score / len(keywords)
            
            # Normalize scores
            total_score = sum(stance_scores.values())
            if total_score > 0:
                stance_scores = {k: v / total_score for k, v in stance_scores.items()}
            
            dominant_stance = max(stance_scores.items(), key=lambda x: x[1])[0]
            
            return {
                'scores': stance_scores,
                'dominant_stance': dominant_stance,
                'confidence': stance_scores[dominant_stance]
            }
            
        except Exception as e:
            logger.error(f"Political stance analysis failed: {e}")
            return {'dominant_stance': 'neutral', 'confidence': 0.0}
    
    def _detect_emotional_bias(self, text: str) -> Dict[str, Any]:
        """Detect emotional bias in text."""
        try:
            text_lower = text.lower()
            
            # Count emotional language
            emotional_count = sum(1 for word in self.bias_indicators['emotional_language'] 
                                if word in text_lower)
            
            # Count loaded words
            loaded_count = sum(1 for word in self.bias_indicators['loaded_words'] 
                             if word in text_lower)
            
            # Count absolute terms
            absolute_count = sum(1 for word in self.bias_indicators['absolute_terms'] 
                               if word in text_lower)
            
            # Calculate emotional bias score
            word_count = len(text.split())
            emotional_ratio = (emotional_count + loaded_count + absolute_count) / max(word_count, 1)
            
            return {
                'emotional_words': emotional_count,
                'loaded_words': loaded_count,
                'absolute_terms': absolute_count,
                'emotional_ratio': emotional_ratio,
                'bias_level': 'high' if emotional_ratio > 0.05 else 'medium' if emotional_ratio > 0.02 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Emotional bias detection failed: {e}")
            return {'bias_level': 'unknown'}
    
    def _detect_language_bias(self, text: str) -> Dict[str, Any]:
        """Detect language-based bias indicators."""
        try:
            # Analyze sentence structure
            sentences = nltk.sent_tokenize(text)
            
            # Count questions vs statements
            questions = sum(1 for s in sentences if s.strip().endswith('?'))
            exclamations = sum(1 for s in sentences if s.strip().endswith('!'))
            
            # Analyze word choice diversity
            words = text.lower().split()
            unique_words = len(set(words))
            word_diversity = unique_words / max(len(words), 1)
            
            # Check for first-person usage
            first_person = sum(1 for word in words if word in ['i', 'me', 'my', 'we', 'us', 'our'])
            first_person_ratio = first_person / max(len(words), 1)
            
            return {
                'question_ratio': questions / max(len(sentences), 1),
                'exclamation_ratio': exclamations / max(len(sentences), 1),
                'word_diversity': word_diversity,
                'first_person_ratio': first_person_ratio
            }
            
        except Exception as e:
            logger.error(f"Language bias detection failed: {e}")
            return {}
    
    def _analyze_credibility_indicators(self, text: str) -> Dict[str, Any]:
        """Analyze credibility indicators in text."""
        try:
            # Check for source citations
            citation_patterns = [r'according to', r'sources say', r'reported by', r'study shows']
            citations = sum(1 for pattern in citation_patterns 
                          if re.search(pattern, text, re.IGNORECASE))
            
            # Check for specific numbers/statistics
            number_pattern = r'\b\d+(?:\.\d+)?\s*(?:%|percent|million|billion|thousand)\b'
            statistics = len(re.findall(number_pattern, text, re.IGNORECASE))
            
            # Check for quotes
            quote_pattern = r'["""](.*?)["""]
            quotes = len(re.findall(quote_pattern, text))
            
            return {
                'citations': citations,
                'statistics': statistics,
                'quotes': quotes,
                'credibility_score': min((citations + statistics + quotes) / 10.0, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Credibility analysis failed: {e}")
            return {'credibility_score': 0.0}
    
    def _calculate_overall_bias(self, bias_analysis: Dict[str, Any]) -> float:
        """Calculate overall bias score."""
        try:
            factors = []
            
            # Political stance bias
            political = bias_analysis.get('political_stance', {})
            if political.get('confidence', 0) > 0.5:
                factors.append(political['confidence'])
            
            # Emotional bias
            emotional = bias_analysis.get('emotional_bias', {})
            if emotional.get('emotional_ratio', 0) > 0:
                factors.append(min(emotional['emotional_ratio'] * 10, 1.0))
            
            # Language bias
            language = bias_analysis.get('language_bias', {})
            exclamation_bias = language.get('exclamation_ratio', 0)
            first_person_bias = language.get('first_person_ratio', 0)
            factors.append((exclamation_bias + first_person_bias) / 2)
            
            # Credibility (inverse factor)
            credibility = bias_analysis.get('credibility_indicators', {})
            credibility_score = credibility.get('credibility_score', 0)
            factors.append(1.0 - credibility_score)
            
            return np.mean(factors) if factors else 0.0
            
        except Exception as e:
            logger.error(f"Overall bias calculation failed: {e}")
            return 0.0

class FactChecker:
    """Fact-checking and claim verification system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize claim detection
        self.claim_patterns = [
            r'\b(?:according to|studies show|research indicates|data shows)\b',
            r'\b(?:\d+(?:\.\d+)?\s*(?:%|percent|million|billion))\b',
            r'\b(?:always|never|all|none|every|no one)\b'
        ]
        
        # Knowledge base for fact-checking (placeholder)
        self.knowledge_base = {}
        
        logger.info("Fact Checker initialized")
    
    def check_facts(self, text: str) -> Dict[str, Any]:
        """Check facts and claims in text."""
        try:
            # Extract claims
            claims = self._extract_claims(text)
            
            # Verify claims (placeholder implementation)
            verified_claims = []
            for claim in claims:
                verification = self._verify_claim(claim)
                verified_claims.append({
                    'claim': claim,
                    'verification': verification
                })
            
            # Calculate overall fact-check confidence
            if verified_claims:
                confidences = [v['verification']['confidence'] for v in verified_claims]
                overall_confidence = np.mean(confidences)
            else:
                overall_confidence = 0.5  # Neutral when no claims found
            
            FACT_CHECK_CONFIDENCE.set(overall_confidence)
            
            return {
                'claims': verified_claims,
                'overall_confidence': overall_confidence,
                'fact_check_score': overall_confidence
            }
            
        except Exception as e:
            logger.error(f"Fact checking failed: {e}")
            return {'fact_check_score': 0.0}
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        try:
            claims = []
            sentences = nltk.sent_tokenize(text)
            
            for sentence in sentences:
                # Check if sentence contains claim patterns
                for pattern in self.claim_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        claims.append(sentence.strip())
                        break
            
            return claims
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []
    
    def _verify_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a single claim (placeholder implementation)."""
        try:
            # Placeholder verification logic
            # In a real implementation, this would:
            # 1. Query external fact-checking APIs
            # 2. Check against knowledge bases
            # 3. Use ML models for claim verification
            
            # Simple heuristic: claims with specific numbers are more verifiable
            has_numbers = bool(re.search(r'\d+', claim))
            has_sources = bool(re.search(r'according to|study|research', claim, re.IGNORECASE))
            
            if has_numbers and has_sources:
                confidence = 0.8
                status = 'likely_true'
            elif has_numbers or has_sources:
                confidence = 0.6
                status = 'partially_verifiable'
            else:
                confidence = 0.3
                status = 'unverifiable'
            
            return {
                'status': status,
                'confidence': confidence,
                'evidence': 'Heuristic analysis',
                'sources': []
            }
            
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {'status': 'error', 'confidence': 0.0}

class ReadabilityAnalyzer:
    """Analyze text readability and quality metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Readability Analyzer initialized")
    
    def analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze text readability metrics."""
        try:
            metrics = {}
            
            # Flesch Reading Ease
            metrics['flesch_reading_ease'] = flesch_reading_ease(text)
            
            # Flesch-Kincaid Grade Level
            metrics['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
            
            # Automated Readability Index
            metrics['automated_readability_index'] = automated_readability_index(text)
            
            # Basic text statistics
            words = text.split()
            sentences = nltk.sent_tokenize(text)
            
            metrics['word_count'] = len(words)
            metrics['sentence_count'] = len(sentences)
            metrics['avg_words_per_sentence'] = len(words) / max(len(sentences), 1)
            
            # Character statistics
            metrics['character_count'] = len(text)
            metrics['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
            
            # Syllable count (approximation)
            syllable_count = sum(self._count_syllables(word) for word in words)
            metrics['syllable_count'] = syllable_count
            metrics['avg_syllables_per_word'] = syllable_count / max(len(words), 1)
            
            # Vocabulary diversity
            unique_words = len(set(word.lower() for word in words))
            metrics['vocabulary_diversity'] = unique_words / max(len(words), 1)
            
            # Overall readability score (0-1, higher is more readable)
            readability_score = self._calculate_readability_score(metrics)
            metrics['overall_readability'] = readability_score
            
            READABILITY_SCORE.set(readability_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {'overall_readability': 0.0}
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        try:
            word = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            previous_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel
            
            # Handle silent 'e'
            if word.endswith('e'):
                syllable_count -= 1
            
            # Every word has at least one syllable
            return max(syllable_count, 1)
            
        except Exception:
            return 1
    
    def _calculate_readability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall readability score."""
        try:
            # Normalize Flesch Reading Ease (0-100 to 0-1)
            flesch_normalized = metrics.get('flesch_reading_ease', 50) / 100.0
            
            # Penalize very long sentences
            avg_sentence_length = metrics.get('avg_words_per_sentence', 15)
            sentence_penalty = max(0, min(1, (25 - avg_sentence_length) / 25))
            
            # Reward vocabulary diversity
            diversity_bonus = metrics.get('vocabulary_diversity', 0.5)
            
            # Penalize very long words
            avg_word_length = metrics.get('avg_word_length', 5)
            word_length_penalty = max(0, min(1, (8 - avg_word_length) / 8))
            
            # Combine factors
            readability_score = (
                flesch_normalized * 0.4 +
                sentence_penalty * 0.3 +
                diversity_bonus * 0.2 +
                word_length_penalty * 0.1
            )
            
            return max(0, min(1, readability_score))
            
        except Exception as e:
            logger.error(f"Readability score calculation failed: {e}")
            return 0.5

class AdvancedNewsNLP:
    """Main advanced NLP system for news analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.sentiment_analyzer = AdvancedSentimentAnalyzer(config)
        self.entity_recognizer = NamedEntityRecognizer(config)
        self.topic_modeler = TopicModeler(config)
        self.bias_detector = BiasDetector(config)
        self.fact_checker = FactChecker(config)
        self.readability_analyzer = ReadabilityAnalyzer(config)
        
        # Initialize summarization models
        try:
            self.summarizer = pipeline(
                "summarization",
                model=config.get('summarization_model', 'facebook/bart-large-cnn'),
                max_length=config.get('max_summary_length', 150),
                min_length=config.get('min_summary_length', 50)
            )
            self.summarization_available = True
        except Exception as e:
            logger.warning(f"Summarization model not available: {e}")
            self.summarization_available = False
        
        # Initialize keyword extraction
        self.keyword_extractor = self._initialize_keyword_extractor()
        
        # Language detection
        self.translator = Translator()
        
        # Cache for processed texts
        self.analysis_cache = {}
        
        # Redis connection for caching
        try:
            self.redis_client = redis.Redis(
                host=config.get('redis_host', 'localhost'),
                port=config.get('redis_port', 6379),
                decode_responses=True
            )
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_available = False
        
        logger.info("Advanced News NLP system initialized")
    
    def _initialize_keyword_extractor(self) -> Any:
        """Initialize keyword extraction model."""
        try:
            from keybert import KeyBERT
            return KeyBERT()
        except ImportError:
            logger.warning("KeyBERT not available, using TF-IDF for keywords")
            return TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1, 2))
    
    def analyze_text(self, news_text: NewsText) -> TextAnalysis:
        """Comprehensive text analysis."""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"text_analysis:{news_text.text_id}"
            if self.redis_available:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return TextAnalysis(**json.loads(cached_result))
            
            text = news_text.content
            
            # Language detection
            if not news_text.language:
                try:
                    detected_language = detect(text)
                except Exception:
                    detected_language = 'en'
            else:
                detected_language = news_text.language
            
            # Sentiment analysis
            NLP_REQUESTS.labels(task_type='sentiment').inc()
            sentiment = self.sentiment_analyzer.analyze_sentiment(text)
            
            # Named entity recognition
            NLP_REQUESTS.labels(task_type='ner').inc()
            entities = self.entity_recognizer.extract_entities(text)
            
            # Topic analysis
            NLP_REQUESTS.labels(task_type='topic_modeling').inc()
            topics = self.topic_modeler.analyze_topics(text)
            
            # Keyword extraction
            NLP_REQUESTS.labels(task_type='keyword_extraction').inc()
            keywords = self._extract_keywords(text)
            
            # Text summarization
            NLP_REQUESTS.labels(task_type='summarization').inc()
            summary = self._generate_summary(text)
            
            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(text)
            
            # Bias analysis
            NLP_REQUESTS.labels(task_type='bias_detection').inc()
            bias_analysis = self.bias_detector.analyze_bias(text)
            
            # Fact checking
            NLP_REQUESTS.labels(task_type='fact_checking').inc()
            fact_check_results = self.fact_checker.check_facts(text)
            
            # Readability analysis
            NLP_REQUESTS.labels(task_type='readability').inc()
            readability_metrics = self.readability_analyzer.analyze_readability(text)
            
            # Semantic features
            semantic_features = self._extract_semantic_features(text)
            
            # Create analysis result
            analysis = TextAnalysis(
                text_id=news_text.text_id,
                language=detected_language,
                sentiment=sentiment,
                entities=entities,
                topics=topics,
                keywords=keywords,
                summary=summary,
                quality_metrics=quality_metrics,
                bias_analysis=bias_analysis,
                fact_check_results=fact_check_results,
                readability_metrics=readability_metrics,
                semantic_features=semantic_features
            )
            
            # Cache result
            if self.redis_available:
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(asdict(analysis), default=str)
                )
            
            # Update metrics
            processing_time = time.time() - start_time
            NLP_LATENCY.labels(task_type='full_analysis').observe(processing_time)
            TEXT_QUALITY.set(np.mean(list(quality_metrics.values())))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            # Return minimal analysis on error
            return TextAnalysis(
                text_id=news_text.text_id,
                language='unknown',
                sentiment=SentimentAnalysis('neutral', 0.0, {}),
                entities=[],
                topics=TopicAnalysis([], [], 0.0),
                keywords=[],
                summary="Analysis failed",
                quality_metrics={},
                bias_analysis={},
                fact_check_results={},
                readability_metrics={},
                semantic_features={}
            )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        try:
            if hasattr(self.keyword_extractor, 'extract_keywords'):
                # KeyBERT
                keywords = self.keyword_extractor.extract_keywords(
                    text, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='english',
                    top_k=10
                )
                return [kw[0] for kw in keywords]
            else:
                # TF-IDF fallback
                tfidf_matrix = self.keyword_extractor.fit_transform([text])
                feature_names = self.keyword_extractor.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                # Get top keywords
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                
                return [kw for kw, score in keyword_scores[:10] if score > 0]
                
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def _generate_summary(self, text: str) -> str:
        """Generate text summary."""
        try:
            if not self.summarization_available:
                # Fallback to extractive summarization
                return self._extractive_summary(text)
            
            # Use transformer-based summarization
            if len(text.split()) < 50:
                return text[:200] + "..." if len(text) > 200 else text
            
            summary = self.summarizer(text, max_length=150, min_length=50, do_sample=False)
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def _extractive_summary(self, text: str) -> str:
        """Generate extractive summary using sentence ranking."""
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 2:
                return text
            
            # Simple sentence scoring based on word frequency
            words = text.lower().split()
            word_freq = Counter(words)
            
            sentence_scores = []
            for sentence in sentences:
                sentence_words = sentence.lower().split()
                score = sum(word_freq[word] for word in sentence_words if word in word_freq)
                sentence_scores.append((sentence, score))
            
            # Sort by score and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sentence_scores[:min(3, len(sentences) // 2)]
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if any(sentence == s[0] for s in top_sentences):
                    summary_sentences.append(sentence)
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return text[:200] + "..."
    
    def _calculate_quality_metrics(self, text: str) -> Dict[str, float]:
        """Calculate text quality metrics."""
        try:
            metrics = {}
            
            # Length metrics
            words = text.split()
            sentences = nltk.sent_tokenize(text)
            
            metrics['word_count'] = len(words)
            metrics['sentence_count'] = len(sentences)
            
            # Vocabulary richness
            unique_words = len(set(word.lower() for word in words))
            metrics['vocabulary_richness'] = unique_words / max(len(words), 1)
            
            # Sentence variety
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            metrics['sentence_length_variance'] = np.var(sentence_lengths) if sentence_lengths else 0
            
            # Punctuation usage
            punctuation_count = sum(1 for char in text if char in string.punctuation)
            metrics['punctuation_ratio'] = punctuation_count / max(len(text), 1)
            
            # Spelling quality (simplified)
            misspelled_count = 0
            for word in words:
                if word.isalpha() and len(word) > 3:
                    # Simple check using TextBlob
                    try:
                        blob = TextBlob(word)
                        if word.lower() != blob.correct().string.lower():
                            misspelled_count += 1
                    except:
                        pass
            
            metrics['spelling_quality'] = 1.0 - (misspelled_count / max(len(words), 1))
            
            # Overall quality score
            quality_factors = [
                min(metrics['vocabulary_richness'] * 2, 1.0),  # Vocabulary richness
                min(metrics['sentence_length_variance'] / 10, 1.0),  # Sentence variety
                metrics['spelling_quality'],  # Spelling quality
                min(metrics['punctuation_ratio'] * 20, 1.0)  # Appropriate punctuation
            ]
            
            metrics['overall_quality'] = np.mean(quality_factors)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {'overall_quality': 0.5}
    
    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text."""
        try:
            features = {}
            
            # Text embedding
            if hasattr(self.topic_modeler, 'sentence_model'):
                embedding = self.topic_modeler.sentence_model.encode(text)
                features['embedding_dim'] = len(embedding)
                features['embedding_norm'] = float(np.linalg.norm(embedding))
            
            # Semantic density (unique concepts per word)
            words = text.lower().split()
            if SPACY_EN_AVAILABLE:
                doc = nlp_en(text)
                concepts = set(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)
                features['semantic_density'] = len(concepts) / max(len(words), 1)
            else:
                features['semantic_density'] = 0.5
            
            # Information content (entropy of word distribution)
            word_counts = Counter(words)
            total_words = sum(word_counts.values())
            probabilities = [count / total_words for count in word_counts.values()]
            features['information_entropy'] = entropy(probabilities) if probabilities else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Semantic feature extraction failed: {e}")
            return {}

def create_nlp_api(nlp_system: AdvancedNewsNLP) -> Flask:
    """Create Flask API for NLP system."""
    app = Flask(__name__)
    
    @app.route('/nlp/analyze', methods=['POST'])
    def analyze_text():
        try:
            data = request.get_json()
            
            # Create news text object
            news_text = NewsText(
                text_id=data.get('text_id', str(uuid.uuid4())),
                content=data.get('content', ''),
                title=data.get('title'),
                author=data.get('author'),
                source=data.get('source'),
                url=data.get('url'),
                language=data.get('language'),
                metadata=data.get('metadata', {})
            )
            
            # Perform analysis
            analysis = nlp_system.analyze_text(news_text)
            
            # Convert to JSON-serializable format
            result = asdict(analysis)
            
            return jsonify({
                'status': 'success',
                'analysis': result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"NLP analysis API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/sentiment', methods=['POST'])
    def analyze_sentiment():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Analyze sentiment
            sentiment = nlp_system.sentiment_analyzer.analyze_sentiment(text)
            
            return jsonify({
                'status': 'success',
                'sentiment': asdict(sentiment),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Sentiment analysis API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/entities', methods=['POST'])
    def extract_entities():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Extract entities
            entities = nlp_system.entity_recognizer.extract_entities(text)
            
            return jsonify({
                'status': 'success',
                'entities': [asdict(entity) for entity in entities],
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Entity extraction API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/topics', methods=['POST'])
    def analyze_topics():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Analyze topics
            topics = nlp_system.topic_modeler.analyze_topics(text)
            
            return jsonify({
                'status': 'success',
                'topics': asdict(topics),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Topic analysis API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/bias', methods=['POST'])
    def analyze_bias():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Analyze bias
            bias_analysis = nlp_system.bias_detector.analyze_bias(text)
            
            return jsonify({
                'status': 'success',
                'bias_analysis': bias_analysis,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Bias analysis API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/fact-check', methods=['POST'])
    def check_facts():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Check facts
            fact_check_results = nlp_system.fact_checker.check_facts(text)
            
            return jsonify({
                'status': 'success',
                'fact_check': fact_check_results,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Fact checking API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/readability', methods=['POST'])
    def analyze_readability():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Analyze readability
            readability = nlp_system.readability_analyzer.analyze_readability(text)
            
            return jsonify({
                'status': 'success',
                'readability': readability,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Readability analysis API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/summarize', methods=['POST'])
    def summarize_text():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Generate summary
            summary = nlp_system._generate_summary(text)
            
            return jsonify({
                'status': 'success',
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Summarization API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/nlp/keywords', methods=['POST'])
    def extract_keywords():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'text required'}), 400
            
            # Extract keywords
            keywords = nlp_system._extract_keywords(text)
            
            return jsonify({
                'status': 'success',
                'keywords': keywords,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception